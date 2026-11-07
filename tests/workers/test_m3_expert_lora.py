"""M3 expert LoRA activation, non-mutating merge and checkpoint-layout tests."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from veomni.lora.moe_layers import LoraIndependentExperts, LoraSharedExperts

from verl.workers.engine.veomni.m3_lora import (
    _merged_m3_weight,
    adapters_disabled,
    merged_m3_lora_weights,
    wrap_m3_lora,
)
from verl.workers.engine.veomni.m3_weight_export import iter_m3_param


@pytest.mark.parametrize("experts", [None, 3])
@pytest.mark.parametrize("zero_b", [False, True])
def test_merged_forward_chain_rule_and_frozen_base(experts, zero_b):
    """Direct-weight merge retains FP32 master gradients under BF16 autocast."""
    torch.manual_seed(461)
    prefix = () if experts is None else (experts,)
    base = torch.randn(*prefix, 24, 32).bfloat16()
    saved = base.clone()
    a = (torch.randn(*prefix, 4, 32) * 0.1).requires_grad_()
    b = (torch.randn(*prefix, 24, 4) * 0.1).requires_grad_()
    if zero_b:
        with torch.no_grad():
            b.zero_()
    x = torch.randn(*prefix, 7, 32).bfloat16().requires_grad_()
    scale = 1.5

    def linear(xx, ww):
        return F.linear(xx, ww) if experts is None else torch.bmm(xx, ww.transpose(-1, -2))

    for step in range(2):
        a.grad = b.grad = x.grad = None
        expected_weight = (base.float() + (b.detach() @ a.detach()) * scale).bfloat16()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            merged = _merged_m3_weight(base, a, b, scale)
            y = linear(x, merged)
        torch.testing.assert_close(merged, expected_weight, rtol=0, atol=0)
        leaf = merged.detach().clone().requires_grad_()
        xr = x.detach().clone().requires_grad_()
        ref = linear(xr, leaf)
        upstream = torch.randn_like(y)
        y.backward(upstream)
        ref.backward(upstream)
        expected_a = b.detach().transpose(-1, -2) @ (leaf.grad.float() * scale)
        expected_b = (leaf.grad.float() * scale) @ a.detach().transpose(-1, -2)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        torch.testing.assert_close(a.grad, expected_a, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(b.grad, expected_b, rtol=1e-6, atol=1e-6)
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        assert (a.grad.norm() == 0) if zero_b and step == 0 else (a.grad.norm() > 0)
        assert b.grad.norm() > 0 and base.grad is None
        torch.testing.assert_close(base, saved, rtol=0, atol=0)
        with torch.no_grad():
            a.add_(a.grad, alpha=-0.01)
            b.add_(b.grad, alpha=-0.01)


def test_merged_forward_rejects_rounded_adapter_masters():
    base = torch.zeros(8, 8, dtype=torch.bfloat16)
    a = torch.zeros(4, 8, dtype=torch.bfloat16)
    b = torch.zeros(8, 4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="FP32 adapter owners"):
        _merged_m3_weight(base, a, b, 1.0)


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
def test_expert_backend_binding(monkeypatch, mode):
    from veomni.lora import ops

    calls = []
    monkeypatch.setattr(ops, "bind_lora_moe_kernels", calls.append)
    wrap_m3_lora(ToyM3(), 4, 4, "all-linear", moe_mode=mode, moe_implementation="fused_npu")
    assert calls == ["npu"]
    wrap_m3_lora(ToyM3(), 4, 4, "all-linear", moe_mode=mode)
    assert calls == ["npu", "eager"]
    wrap_m3_lora(ToyM3(), 4, 4, ["q_proj"], moe_mode=mode, moe_implementation="fused_npu")
    assert calls == ["npu", "eager"]
    with pytest.raises(ValueError, match="only eager or fused_npu"):
        wrap_m3_lora(ToyM3(), 4, 4, "all-linear", moe_mode=mode, moe_implementation="fused_triton")


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_npu_forward_backward(mode, monkeypatch):
    """Run only when explicitly requested, to avoid using a busy training NPU."""
    import os

    if os.environ.get("M31_TEST_NPU") != "1":
        pytest.skip("Set M31_TEST_NPU=1 on a free device")
    from veomni.lora import ops

    # Exercise the production wrapper and its dispatch, starting unbound.
    monkeypatch.setattr(ops, "_fused_lora_moe_forward", None)
    monkeypatch.setattr(ops, "_fused_independent_lora_moe_forward", None)
    wrapped = wrap_m3_lora(ToyM3(), 4, 7, "all-linear", moe_mode=mode, moe_implementation="fused_npu")
    m = next(m for m in wrapped.modules() if isinstance(m, (LoraSharedExperts, LoraIndependentExperts)))
    m = m.to("npu", dtype=torch.bfloat16)
    for name, p in m.named_parameters():
        if ".lora_B." in name:
            with torch.no_grad():
                p.normal_(0, 0.03)
    x = torch.randn(16, 8, device="npu", dtype=torch.bfloat16, requires_grad=True)
    ids = torch.arange(16, device="npu").remainder(4).reshape(-1, 1)
    weights = torch.ones(16, 1, device="npu", dtype=torch.bfloat16)
    y = m(x, ids, weights)
    ref = m._eager_forward(x, ids, weights)
    torch.testing.assert_close(y, ref, atol=0.08, rtol=0.05)
    parameters = [x] + [p for p in m.parameters() if p.requires_grad]
    a = torch.autograd.grad(y.float().sum(), parameters)
    b = torch.autograd.grad(ref.float().sum(), parameters)
    for ga, gb in zip(a, b, strict=True):
        torch.testing.assert_close(ga, gb, atol=0.15, rtol=0.1)


class DtypeRouter(nn.Module):
    def __init__(self, parameter_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.full((4, 8), 0.123456))
        bias = torch.full((4,), 0.123456)
        if parameter_bias:
            self.e_score_correction_bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_buffer("e_score_correction_bias", bias)

    def forward(self, x):
        return F.linear(x.to(self.weight.dtype), self.weight) + self.e_score_correction_bias


@pytest.mark.parametrize("mode", ["shared", "independent"])
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("parameter_bias", [False, True])
def test_m3_storage_precision(mode, device, parameter_bias):
    from verl.workers.engine.veomni.m3_lora import prepare_m3_lora_dtypes

    with torch.device(device):
        toy = ToyM3()
        toy.model.language_model.layers[0].mlp.gate = DtypeRouter(parameter_bias)
        toy.register_buffer("indices", torch.arange(4))
        toy.register_buffer("rotary", torch.ones(4, dtype=torch.float32))
    model = wrap_m3_lora(toy, 4, 7, "all-linear", moe_mode=mode)
    model.get_ignore_modules_in_mixed_precision = lambda: (nn.LayerNorm,)
    before = {n: (id(p), p.requires_grad, p.detach().clone()) for n, p in model.named_parameters()}
    prepare_m3_lora_dtypes(model)
    prepare_m3_lora_dtypes(model)  # Idempotent and keeps pre-existing exclusions.
    assert model.get_ignore_modules_in_mixed_precision() == (nn.LayerNorm, DtypeRouter)
    for name, p in model.named_parameters():
        identity, trainable, value = before[name]
        assert id(p) == identity and p.requires_grad == trainable
        fp32 = trainable or name.endswith(".mlp.gate.weight") or name.endswith(".e_score_correction_bias")
        assert p.dtype == (torch.float32 if fp32 else torch.bfloat16), name
        if device == "cpu":
            torch.testing.assert_close(p, value.to(p.dtype), rtol=0, atol=0)
    for name, buffer in model.named_buffers():
        if name.endswith("e_score_correction_bias") or name.endswith("rotary"):
            assert buffer.dtype == torch.float32
        if name.endswith("indices"):
            assert buffer.dtype == torch.int64


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_mixed_storage_merge_and_gradients(mode):
    from verl.workers.engine.veomni.m3_lora import prepare_m3_lora_dtypes

    torch.manual_seed(123)
    model = wrap_m3_lora(ToyM3(), 4, 7, "all-linear", moe_mode=mode)
    prepare_m3_lora_dtypes(model)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_B." in name:
                p.normal_(0, 0.03)
    before = {n: t.detach().clone() for n, t in model.state_dict().items()}
    merged = dict(merged_m3_lora_weights(model, model.state_dict(), "cpu").items())
    expert = next(m for m in model.modules() if isinstance(m, (LoraSharedExperts, LoraIndependentExperts)))
    x = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.arange(16).remainder(4).reshape(-1, 1)
    weights = torch.ones(16, 1, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expert._eager_forward(x, ids, weights).float().square().mean().backward()
    for name, p in expert.named_parameters():
        if p.requires_grad:
            assert p.dtype == p.grad.dtype == torch.float32
            assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
        else:
            assert p.dtype == torch.bfloat16 and p.grad is None
    root = "model.language_model.layers.0.mlp.experts"
    for projection, specs in (("gate_up_proj", ("gate_proj", "up_proj")), ("down_proj", ("down_proj",))):
        base = getattr(expert, projection).base_layer.weight.detach().float()
        updates = []
        for spec in specs:
            a, b = expert.get_lora_A_weight(spec), expert.get_lora_B_weight(spec)
            updates.append(b @ a * expert._lora_scale_value)
        delta = torch.cat(updates, dim=-2) if len(updates) > 1 else updates[0]
        torch.testing.assert_close(merged[f"{root}.{projection}"], (base + delta).bfloat16(), rtol=0, atol=0)
    for name, t in model.state_dict().items():
        torch.testing.assert_close(t, before[name], rtol=0, atol=0)
