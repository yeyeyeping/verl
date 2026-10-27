# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Stream MiniMax M3 VeOmni parameters in the public checkpoint layout.

VeOmni's ordinary EP plan slices experts before FSDP wraps them. Consequently
DTensor.full_tensor() restores an EP-local stack, not necessarily all experts.
Also accept DTensors whose mesh already restores the entire expert dimension.
"""

import re

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def _check_shape(name, tensor, shape):
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"M3 export {name}: expected shape {tuple(shape)}, got {tuple(tensor.shape)}")


def export_m3_tensor(name, tensor, config, expert_id_base=0):
    """Validate and invert the public-to-VeOmni checkpoint conversion."""
    text = config.text_config
    hidden = text.hidden_size
    if name in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
        _check_shape(name, tensor, (text.vocab_size, hidden))
        target = (
            "language_model.lm_head.weight" if name == "lm_head.weight" else "language_model.model.embed_tokens.weight"
        )
    elif name == "model.language_model.norm.weight":
        _check_shape(name, tensor, (hidden,))
        target = "language_model.model.norm.weight"
    elif match := re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", name):
        layer, suffix = int(match[1]), match[2]
        if not 0 <= layer < text.num_hidden_layers:
            raise ValueError(f"M3 export: layer outside config: {name}")
        prefix = f"language_model.model.layers.{layer}."
        head_dim = getattr(text, "head_dim", hidden // text.num_attention_heads)
        attn_shapes = {
            "q_proj.weight": (text.num_attention_heads * head_dim, hidden),
            "k_proj.weight": (text.num_key_value_heads * head_dim, hidden),
            "v_proj.weight": (text.num_key_value_heads * head_dim, hidden),
            "o_proj.weight": (hidden, text.num_attention_heads * head_dim),
            "q_norm.weight": (head_dim,),
            "k_norm.weight": (head_dim,),
        }
        if suffix.startswith("mlp.experts."):
            count = tensor.shape[0] if tensor.ndim == 3 else 0
            if count <= 0 or expert_id_base < 0 or expert_id_base + count > text.num_local_experts:
                raise ValueError(f"M3 export: invalid expert range for {name}: {expert_id_base}+{count}")
            inter = text.intermediate_size
            proj = suffix.removeprefix("mlp.experts.")
            if proj == "gate_up_proj":
                _check_shape(name, tensor, (count, 2 * inter, hidden))
                for i in range(count):
                    base = prefix + f"block_sparse_moe.experts.{expert_id_base + i}."
                    yield base + "w1.weight", tensor[i, :inter].contiguous()
                    yield base + "w3.weight", tensor[i, inter:].contiguous()
            elif proj == "down_proj":
                _check_shape(name, tensor, (count, hidden, inter))
                for i in range(count):
                    yield prefix + f"block_sparse_moe.experts.{expert_id_base + i}.w2.weight", tensor[i].contiguous()
            else:
                raise ValueError(f"M3 export: unsupported expert parameter {name}")
            return
        if suffix.startswith("mlp.shared_experts.") or suffix in ("mlp.gate_up_proj.weight", "mlp.down_proj.weight"):
            shared = suffix.startswith("mlp.shared_experts.")
            inter = text.shared_intermediate_size if shared else text.dense_intermediate_size
            mlp_prefix = prefix + ("block_sparse_moe.shared_experts." if shared else "mlp.")
            if suffix.endswith("gate_up_proj.weight"):
                _check_shape(name, tensor, (2 * inter, hidden))
                yield mlp_prefix + "gate_proj.weight", tensor[:inter].contiguous()
                yield mlp_prefix + "up_proj.weight", tensor[inter:].contiguous()
                return
            if not suffix.endswith("down_proj.weight"):
                raise ValueError(f"M3 export: unsupported MLP parameter {name}")
            _check_shape(name, tensor, (hidden, inter))
            target = mlp_prefix + "down_proj.weight"
        elif suffix == "mlp.gate.weight":
            _check_shape(name, tensor, (text.num_local_experts, hidden))
            target = prefix + "block_sparse_moe.gate.weight"
        elif suffix == "mlp.gate.e_score_correction_bias":
            _check_shape(name, tensor, (text.num_local_experts,))
            target = prefix + "block_sparse_moe.e_score_correction_bias"
        elif suffix in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            _check_shape(name, tensor, (hidden,))
            target = prefix + suffix
        elif suffix.startswith("self_attn.indexer."):
            part = suffix.removeprefix("self_attn.indexer.")
            dim = text.index_head_dim
            shapes = {
                "q_proj.weight": (text.index_n_heads * dim, hidden),
                "k_proj.weight": (dim, hidden),
                "q_norm.weight": (dim,),
                "k_norm.weight": (dim,),
            }
            if part not in shapes:
                raise ValueError(f"M3 export: unsupported indexer parameter {name}")
            _check_shape(name, tensor, shapes[part])
            target = prefix + "self_attn.index_" + part
        elif suffix.startswith("self_attn.") and suffix.removeprefix("self_attn.") in attn_shapes:
            _check_shape(name, tensor, attn_shapes[suffix.removeprefix("self_attn.")])
            target = prefix + suffix
        else:
            raise ValueError(f"M3 export: unsupported language parameter {name}")
    elif name.startswith("model.vision_tower."):
        vision = config.vision_config
        dim, inter = vision.hidden_size, vision.intermediate_size
        suffix = name.removeprefix("model.vision_tower.")
        target = "vision_tower.vision_model." + suffix
        if suffix == "embeddings.proj.weight":
            _check_shape(
                name,
                tensor,
                (dim, vision.num_channels, vision.temporal_patch_size, vision.patch_size, vision.patch_size),
            )
            target = "vision_tower.vision_model.embeddings.patch_embedding.weight"
        elif suffix in ("pre_layrnorm.weight", "pre_layrnorm.bias"):
            _check_shape(name, tensor, (dim,))
        elif match := re.fullmatch(r"layers\.(\d+)\.(.+)\.(weight|bias)", suffix):
            if not 0 <= int(match[1]) < vision.num_hidden_layers:
                raise ValueError(f"M3 export: vision layer outside config: {name}")
            part, kind = match[2], match[3]
            shapes = {
                "self_attn.q_proj": (dim, dim),
                "self_attn.k_proj": (dim, dim),
                "self_attn.v_proj": (dim, dim),
                "self_attn.out_proj": (dim, dim),
                "mlp.fc1": (inter, dim),
                "mlp.fc2": (dim, inter),
                "layer_norm1": (dim,),
                "layer_norm2": (dim,),
            }
            if part not in shapes:
                raise ValueError(f"M3 export: unsupported vision parameter {name}")
            shape = shapes[part]
            _check_shape(name, tensor, shape[:1] if kind == "bias" else shape)
            target = "vision_tower.vision_model.encoder." + suffix
        else:
            raise ValueError(f"M3 export: unsupported vision parameter {name}")
    elif match := re.fullmatch(r"model\.multi_modal_projector\.(merge_)?linear_([12])\.(weight|bias)", name):
        merge, index, kind = match.groups()
        input_size = config.merged_hidden_size if merge else config.vision_config.hidden_size
        shape = (config.projector_hidden_size, input_size) if index == "1" else (hidden, config.projector_hidden_size)
        _check_shape(name, tensor, shape[:1] if kind == "bias" else shape)
        target = ("patch_merge_mlp." if merge else "multi_modal_projector.") + f"linear_{index}.{kind}"
    else:
        raise ValueError(f"M3 export: unsupported parameter {name}; refusing an unconverted weight")
    yield target, tensor.contiguous()


def iter_m3_param(params, config, parallel_state, device):
    """Reconstruct experts in EP-rank order without retaining a global stack."""
    total_experts = config.text_config.num_local_experts
    for name, param in params.items():
        tensor = param.to(device, non_blocking=False).full_tensor() if isinstance(param, DTensor) else param
        expert = ".mlp.experts." in name
        if expert and parallel_state.ep_enabled and tensor.shape[0] != total_experts:
            ep_size = parallel_state.ep_size
            if tensor.ndim != 3 or tensor.shape[0] * ep_size != total_experts:
                raise ValueError(
                    f"M3 export {name}: EP-local shape {tuple(tensor.shape)} cannot reconstruct {total_experts} experts"
                )
            tensor = tensor.to(device, non_blocking=False).contiguous()
            buffer = torch.empty_like(tensor)
            for rank in range(ep_size):
                chunk = tensor if rank == parallel_state.ep_rank else buffer
                dist.broadcast(
                    chunk, src=dist.get_global_rank(parallel_state.ep_group, rank), group=parallel_state.ep_group
                )
                yield from export_m3_tensor(name, chunk, config, rank * chunk.shape[0])
        else:
            if expert and tensor.shape[0] != total_experts:
                raise ValueError(f"M3 export {name}: missing experts without EP reconstruction")
            yield from export_m3_tensor(name, tensor, config)
