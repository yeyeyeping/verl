"""Local two-process CPU FSDP2 precision probe; does not allocate an NPU."""

import torch
import torch.distributed as dist
from test_m3_expert_lora import DtypeRouter, ToyM3
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from veomni.distributed.parallel_state import _init_parallel_state

from verl.workers.engine.veomni.m3_lora import prepare_m3_lora_dtypes, wrap_m3_lora


class ForwardM3(ToyM3):
    def __init__(self):
        super().__init__()
        self.model.language_model.layers[0].mlp.gate = DtypeRouter()

    def forward(self, x):
        layer = self.model.language_model.layers[0]
        x = layer.self_attn.q_proj(x)
        scores = layer.mlp.gate(x).sigmoid()
        ids = torch.arange(x.shape[0]).remainder(4).reshape(-1, 1)
        return layer.mlp.experts(x, ids, scores.gather(1, ids))


def main():
    dist.init_process_group("gloo")
    ps = _init_parallel_state(dp_size=dist.get_world_size(), dp_shard_size=dist.get_world_size(), device_type="cpu")
    mesh = ps.fsdp_mesh
    for mode in ("shared", "independent"):
        torch.manual_seed(71)
        model = wrap_m3_lora(ForwardM3(), 4, 7, "all-linear", moe_mode=mode)
        prepare_m3_lora_dtypes(model)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".lora_B." in name:
                    p.normal_(0, 0.03)
        layer = model.base_model.model.model.language_model.layers[0]
        router_dtypes = []
        layer.mlp.gate.register_forward_pre_hook(
            lambda m, args, observed=router_dtypes: observed.append(m.weight.dtype)
        )
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fully_shard(layer.mlp.gate, mesh=mesh, reshard_after_forward=False)
        fully_shard(layer.mlp.experts, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01, weight_decay=0)
        x = torch.randn(16, 8)
        output = model(x)
        loss = output.float().square().mean()
        loss.backward()
        assert router_dtypes and set(router_dtypes) == {torch.float32}, router_dtypes
        nonzero_expert_grads = 0
        for name, p in model.named_parameters():
            expected = torch.float32 if p.requires_grad or name.endswith(".mlp.gate.weight") else torch.bfloat16
            assert p.dtype == expected, (name, p.dtype)
            if p.requires_grad and ".experts." in name:
                assert p.grad is not None and p.grad.dtype == torch.float32, name
                grad = p.grad.to_local()
                assert torch.isfinite(grad).all() and grad.abs().sum() > 0, name
                nonzero_expert_grads += 1
        assert nonzero_expert_grads == 6
        optimizer.step()
        changed = 0
        for name, p in model.named_parameters():
            before, after = initial[name].to_local(), p.detach().to_local()
            if not p.requires_grad:
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            elif ".experts." in name:
                assert not torch.equal(before, after), name
                changed += 1
        for state in optimizer.state.values():
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        if dist.get_rank() == 0:
            print(
                f"CPU_FSDP2_DTYPE_PASS mode={mode} expert_gradients={nonzero_expert_grads} expert_updates={changed}",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
