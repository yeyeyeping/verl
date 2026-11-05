"""M3 expert LoRA activation, non-mutating merge and checkpoint-layout tests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from veomni.lora.moe_layers import LoraIndependentExperts, LoraSharedExperts

from verl.workers.engine.veomni.m3_lora import (
    adapters_disabled,
    merged_m3_lora_weights,
    wrap_m3_lora,
)
from verl.workers.engine.veomni.m3_weight_export import iter_m3_param


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts, self.hidden_dim, self.intermediate_dim = 4, 8, 12
        self.swiglu_alpha, self.swiglu_limit = 1.702, 7.0
        self.gate_up_proj = nn.Parameter(torch.randn(4, 24, 8) * 0.3)
        self.down_proj = nn.Parameter(torch.randn(4, 8, 12) * 0.3)

    def forward(self, x, ids, weights):
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            for k in range(ids.shape[1]):
                rows = torch.where(ids[:, k] == e)[0]
                gate, up = F.linear(x[rows], self.gate_up_proj[e]).chunk(2, -1)
                gate, up = gate.clamp(max=7), up.clamp(-7, 7)
                mid = (up + 1) * gate * torch.sigmoid(1.702 * gate)
                out.index_add_(0, rows, F.linear(mid, self.down_proj[e]) * weights[rows, k, None])
        return out


class ToyM3(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="minimax_m3_vl",
            text_config=SimpleNamespace(
                num_hidden_layers=1,
                hidden_size=8,
                num_local_experts=4,
                intermediate_size=12,
                dense_intermediate_size=12,
                shared_intermediate_size=12,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=4,
            ),
        )
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.ModuleDict(
            {p: nn.Linear(8, 8, bias=False) for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
        )
        layer.mlp = nn.Module()
        layer.mlp.experts = Experts()
        layer.mlp.shared_experts = nn.ModuleDict(
            {"gate_up_proj": nn.Linear(8, 24, bias=False), "down_proj": nn.Linear(12, 8, bias=False)}
        )
        self.model.language_model.layers = nn.ModuleList([layer])


@pytest.mark.parametrize("mode", ["shared", "independent"])
@pytest.mark.parametrize("rslora", [False, True])
def test_activation_gradients_and_merge(mode, rslora):
    torch.manual_seed(123)
    original = ToyM3()
    reference = deepcopy(original)
    wrapped = wrap_m3_lora(original, 4, 7, "all-linear", moe_mode=mode)
    expert = next(m for m in wrapped.modules() if isinstance(m, (LoraSharedExperts, LoraIndependentExperts)))
    base = reference.model.language_model.layers[0].mlp.experts
    x = (torch.randn(11, 8) * 5).requires_grad_()
    ids = torch.tensor([[i % 4, (i + 1) % 4] for i in range(11)])
    weights = torch.softmax(torch.randn(11, 2), -1)
    torch.testing.assert_close(expert._eager_forward(x, ids, weights), base(x, ids, weights))
    if rslora:
        expert._lora_scale_value = 7 / (4**0.5)
        expert.lora_scaling.fill_(expert._lora_scale_value)
    with torch.no_grad():
        for name, p in wrapped.named_parameters():
            if ".lora_B." in name:
                p.normal_(0, 0.08)
    before = {n: t.detach().clone() for n, t in wrapped.state_dict().items()}
    y = expert._eager_forward(x, ids, weights)
    y.square().mean().backward()
    for name, p in expert.named_parameters():
        if ".lora_" in name:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name
    merged = dict(merged_m3_lora_weights(wrapped, wrapped.state_dict(), "cpu").items())
    reference.load_state_dict(merged, strict=True)
    xx = x.detach().clone().requires_grad_()
    yy = base(xx, ids, weights)
    torch.testing.assert_close(y, yy, rtol=2e-5, atol=5e-5)
    yy.square().mean().backward()
    torch.testing.assert_close(x.grad, xx.grad, rtol=5e-4, atol=1e-4)
    for name, tensor in wrapped.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    exported = dict(
        iter_m3_param(
            merged_m3_lora_weights(wrapped, wrapped.state_dict(), "cpu"),
            wrapped.config,
            SimpleNamespace(ep_enabled=False),
            "cpu",
        )
    )
    for e in range(4):
        prefix = f"language_model.model.layers.0.block_sparse_moe.experts.{e}."
        gu = merged["model.language_model.layers.0.mlp.experts.gate_up_proj"][e]
        torch.testing.assert_close(exported[prefix + "w1.weight"], gu[:12])
        torch.testing.assert_close(exported[prefix + "w3.weight"], gu[12:])
    with pytest.raises(RuntimeError, match="restore"):
        with adapters_disabled(wrapped):
            assert expert._lora_scale_value == 0
            raise RuntimeError("restore")
    assert expert._lora_scale_value != 0
    torch.testing.assert_close(expert.lora_scaling, before[next(n for n in before if n.endswith("lora_scaling"))])


def test_bad_targets_rejected():
    with pytest.raises(ValueError, match="indexer"):
        wrap_m3_lora(ToyM3(), 4, 4, ["indexer.q_proj"])
    with pytest.raises(ValueError, match="expert target"):
        wrap_m3_lora(ToyM3(), 4, 4, target_parameters=["*.weight"])


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_npu_forward_backward(mode):
    """Run only when explicitly requested, to avoid using a busy training NPU."""
    import os

    if os.environ.get("M31_TEST_NPU") != "1":
        pytest.skip("Set M31_TEST_NPU=1 on a free device")
    from veomni.lora.ops.npu_moe_group_gemm import _npu_fused_lora_moe_forward

    cls = LoraSharedExperts if mode == "shared" else LoraIndependentExperts
    m = cls(Experts(), r=4, lora_alpha=7).to("npu", dtype=torch.bfloat16)
    for name, p in m.named_parameters():
        if ".lora_B." in name:
            with torch.no_grad():
                p.normal_(0, 0.03)
    x = torch.randn(16, 8, device="npu", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.arange(16, device="npu").remainder(4).reshape(-1, 1)
    weights = torch.ones(16, 1, device="npu", dtype=torch.bfloat16)
    kwargs = dict(
        num_experts=4,
        routing_weights=weights,
        selected_experts=ids,
        hidden_states=x,
        fc1_1_2_weight=m.gate_up_proj.base_layer.weight,
        fc2_weight=m.down_proj.base_layer.weight,
        independent=mode == "independent",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    for short, spec in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
        kwargs["lora_a_" + short] = m.get_lora_A_weight(spec)
        kwargs["lora_b_" + short] = m.get_lora_B_weight(spec)
        kwargs["lora_scale_" + short] = m._lora_scale_value
    y = _npu_fused_lora_moe_forward(**kwargs)
    ref = m._eager_forward(x, ids, weights)
    torch.testing.assert_close(y, ref, atol=0.08, rtol=0.05)
    parameters = [x] + [p for p in m.parameters() if p.requires_grad]
    a = torch.autograd.grad(y.float().sum(), parameters)
    b = torch.autograd.grad(ref.float().sum(), parameters)
    for ga, gb in zip(a, b, strict=True):
        torch.testing.assert_close(ga, gb, atol=0.15, rtol=0.1)
