# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU regressions for M3 public weights and prompt expansion."""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from verl.workers.engine.veomni import m3_weight_export as export
from verl.workers.rollout.utils import m3_collapse_image_tokens, validate_m3_prompt_alignment


@pytest.fixture
def config():
    return NS(
        text_config=NS(
            hidden_size=4,
            intermediate_size=3,
            dense_intermediate_size=6,
            shared_intermediate_size=3,
            num_local_experts=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            vocab_size=8,
            index_head_dim=2,
            index_n_heads=1,
        ),
        vision_config=NS(
            hidden_size=4, intermediate_size=6, num_hidden_layers=1, num_channels=3, temporal_patch_size=2, patch_size=2
        ),
        projector_hidden_size=6,
        merged_hidden_size=16,
    )


def test_expert_layout_and_dense_split(config):
    name = "model.language_model.layers.1.mlp.experts.gate_up_proj"
    tensor = torch.arange(4 * 6 * 4).reshape(4, 6, 4)
    weights = dict(export.export_m3_tensor(name, tensor, config))
    for expert in range(4):
        prefix = f"language_model.model.layers.1.block_sparse_moe.experts.{expert}"
        torch.testing.assert_close(weights[prefix + ".w1.weight"], tensor[expert, :3])
        torch.testing.assert_close(weights[prefix + ".w3.weight"], tensor[expert, 3:])
    dense = torch.arange(12 * 4).reshape(12, 4)
    values = list(export.export_m3_tensor("model.language_model.layers.0.mlp.gate_up_proj.weight", dense, config))
    torch.testing.assert_close(torch.cat([v for _, v in values]), dense)


@pytest.mark.parametrize("rank", [0, 1])
def test_ep_reconstructs_global_experts(monkeypatch, config, rank):
    full = torch.arange(4 * 4 * 3).reshape(4, 4, 3)
    group = object()
    monkeypatch.setattr(export.dist, "get_global_rank", lambda g, r: r + 10)

    def broadcast(tensor, src, group):
        tensor.copy_(full[(src - 10) * 2 : (src - 9) * 2])

    monkeypatch.setattr(export.dist, "broadcast", broadcast)
    params = {"model.language_model.layers.1.mlp.experts.down_proj": full[rank * 2 : rank * 2 + 2].clone()}
    state = NS(ep_enabled=True, ep_size=2, ep_rank=rank, ep_group=group)
    values = list(export.iter_m3_param(params, config, state, "cpu"))
    assert len(values) == 4
    for index, (name, weight) in enumerate(values):
        assert name.endswith(f"experts.{index}.w2.weight")
        torch.testing.assert_close(weight, full[index])


def test_global_experts_are_not_broadcast_twice(monkeypatch, config):
    monkeypatch.setattr(export.dist, "broadcast", lambda *a, **kw: pytest.fail("already global"))
    params = {"model.language_model.layers.1.mlp.experts.down_proj": torch.zeros(4, 4, 3)}
    assert len(list(export.iter_m3_param(params, config, NS(ep_enabled=True), "cpu"))) == 4


@pytest.mark.parametrize(
    "name,shape,target",
    [
        (
            "model.language_model.layers.1.self_attn.indexer.q_proj.weight",
            (2, 4),
            "language_model.model.layers.1.self_attn.index_q_proj.weight",
        ),
        (
            "model.language_model.layers.1.mlp.gate.e_score_correction_bias",
            (4,),
            "language_model.model.layers.1.block_sparse_moe.e_score_correction_bias",
        ),
        (
            "model.vision_tower.layers.0.mlp.fc1.weight",
            (6, 4),
            "vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight",
        ),
        ("model.multi_modal_projector.merge_linear_1.weight", (6, 16), "patch_merge_mlp.linear_1.weight"),
    ],
)
def test_nonexpert_names_and_shape_validation(config, name, shape, target):
    assert list(export.export_m3_tensor(name, torch.zeros(shape), config))[0][0] == target
    with pytest.raises(ValueError, match="expected shape"):
        list(export.export_m3_tensor(name, torch.zeros(1), config))


def test_unknown_weight_and_bad_expert_count_fail(config):
    with pytest.raises(ValueError, match="unsupported parameter"):
        list(export.export_m3_tensor("new.weight", torch.zeros(1), config))
    with pytest.raises(ValueError, match="cannot reconstruct"):
        list(
            export.iter_m3_param(
                {"model.language_model.layers.1.mlp.experts.down_proj": torch.zeros(3, 4, 3)},
                config,
                NS(ep_enabled=True, ep_size=2),
                "cpu",
            )
        )


@pytest.fixture
def processor():
    return NS(image_token_id=25, vision_start_token_id=21, vision_end_token_id=22)


def test_image_boundaries_and_adjacent_images(processor):
    assert m3_collapse_image_tokens([1, 21, 25, 25, 22, 21, 25, 22, 2], processor, 2) == [1, 25, 25, 2]
    assert m3_collapse_image_tokens([1, 2], processor, 0) == [1, 2]
    assert m3_collapse_image_tokens([25, 25], processor, 2) == [25, 25]


def test_native_m3_processor_preserves_template(monkeypatch):
    import transformers

    from verl.utils.tokenizer import hf_processor

    class MiniMaxM3VLProcessor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = NS(model_type="minimax_m3_vl")
    tokenizer = NS(chat_template="original template")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **kw: pytest.fail("remote processor"))
    monkeypatch.setattr(transformers, "MiniMaxM3VLProcessor", MiniMaxM3VLProcessor)
    for name in ("MiniMaxM3VLImageProcessor", "MiniMaxM3VLVideoProcessor"):
        monkeypatch.setattr(transformers, name, NS(from_pretrained=lambda *a, **kw: object()))
    processor = hf_processor("unused", trust_remote_code=True)
    assert processor.chat_template == tokenizer.chat_template
    assert processor.config is config
    assert not hasattr(processor, "get_rope_index")


@pytest.mark.parametrize("tokens,count", [([21, 25], 1), ([22], 0), ([21, 22], 1), ([21, 26, 22], 0), ([25], 2)])
def test_malformed_images_fail(processor, tokens, count):
    with pytest.raises(ValueError):
        m3_collapse_image_tokens(tokens, processor, count)


def test_alignment_checks_values_not_only_length():
    validate_m3_prompt_alignment([1, 2], [1, 2])
    with pytest.raises(ValueError, match="token 1"):
        validate_m3_prompt_alignment([1, 2], [1, 3])
    with pytest.raises(ValueError):
        validate_m3_prompt_alignment([1, 2], None)


def test_weight_interval_bootstrap_resume_and_version(monkeypatch):
    from verl.checkpoint_engine import base

    manager = object.__new__(base.CheckpointEngineManager)
    manager.backend = "naive"
    manager.update_weights_interval = 3
    manager._weights_initialized = False
    manager._last_weight_step = None
    manager.actor_wg = NS(update_weights=MagicMock(return_value=[]))
    manager.replicas = []
    manager.wake_up_replicas = AsyncMock()
    manager.resume_generation_replicas = AsyncMock()
    monkeypatch.setattr(base.ray, "get", lambda refs: [])
    manager.update_weights(global_steps=1)  # Mandatory seed even on a skipped modulo.
    assert manager.actor_wg.update_weights.call_count == 1
    assert manager.update_weights(global_steps=2) == {"weight_sync/skipped": 1, "weight_sync/weight_step": 1}
    manager.wake_up_replicas.assert_awaited_once()
    manager.resume_generation_replicas.assert_awaited_once()
    assert manager.actor_wg.update_weights.call_count == 1
    manager.update_weights(global_steps=3)
    assert manager._last_weight_step == 3
    manager.add_replicas([object()])
    manager.update_weights(global_steps=4)
    assert manager.actor_wg.update_weights.call_count == 3
    manager.update_weights(global_steps=None)  # Explicit unversioned sync is never skipped.
    assert manager.actor_wg.update_weights.call_count == 4
    assert manager.update_weights(global_steps=5) == {"weight_sync/skipped": 1}


def test_large_shm_tensor_roundtrip(monkeypatch, tmp_path):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    import zmq

    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as transfer

    monkeypatch.setattr(transfer, "get_torch_device", lambda: MagicMock())
    monkeypatch.setattr(transfer, "is_support_ipc", lambda: False)
    address = f"ipc://{tmp_path}/weights.sock"
    sender = transfer.BucketedWeightSender(address, bucket_size_mb=1, use_shm=True)
    receiver = transfer.BucketedWeightReceiver(address, device=torch.device("cpu"), use_shm=True)
    sender.bucket_size = 16  # Exercise chunks without allocating gigabytes.
    for endpoint in (sender, receiver):
        original = endpoint._init_socket

        def init_socket(endpoint=endpoint, original=original):
            original()
            endpoint.socket.setsockopt(zmq.RCVTIMEO, 10000)
            endpoint.socket.setsockopt(zmq.LINGER, 0)

        monkeypatch.setattr(endpoint, "_init_socket", init_socket)
    expected = {
        "small": torch.tensor([1.0, 2.0]),
        "large": torch.arange(35, dtype=torch.float32).reshape(5, 7),
        "tail": torch.tensor([7, 8, 9], dtype=torch.int16),
    }
    actual, final_flags = {}, []

    def receive(weights, is_last):
        actual.update({name: value.clone() for name, value in weights})
        final_flags.append(is_last)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(receiver.receive_weights, receive)
        asyncio.run(sender.async_send_weights(iter(expected.items())))
        future.result(timeout=15)
    assert final_flags.count(True) == 1
    assert final_flags[-1] is True
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
