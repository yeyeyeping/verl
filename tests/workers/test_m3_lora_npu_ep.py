"""Two-NPU EP forward/backward parity against the unsharded eager oracle."""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from test_m3_expert_lora import Experts
from veomni.lora.moe_layers import LoraIndependentExperts, LoraSharedExperts
from veomni.lora.ops.npu_moe_group_gemm import _npu_ep_fused_lora_moe_forward


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    world = dist.get_world_size()
    for mode, cls in (("shared", LoraSharedExperts), ("independent", LoraIndependentExperts)):
        torch.manual_seed(17)
        m = cls(Experts(), r=4, lora_alpha=7)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if ".lora_B." in name:
                    p.normal_(0, 0.03)
        # All ranks use the same reference inputs; each local expert's gradient
        # must equal world times the corresponding unsharded reference gradient.
        cpu_x = torch.randn(16, 8)
        m = m.to(f"npu:{rank}", dtype=torch.bfloat16)
        x = cpu_x.to(f"npu:{rank}", dtype=torch.bfloat16).requires_grad_()
        ids = torch.arange(16, device=x.device).remainder(4).reshape(-1, 1)
        weights = torch.ones(16, 1, device=x.device, dtype=x.dtype)
        ref = m._eager_forward(x, ids, weights)
        ref_params = [p for p in m.parameters() if p.requires_grad]
        ref_grads = torch.autograd.grad(ref.float().sum(), [x] + ref_params)
        pairs = {}
        for name, p in m.named_parameters():
            if ".lora_" in name:
                part = p.chunk(world, 0)[rank] if mode == "independent" else p
                pairs[name] = part.detach().clone().requires_grad_()
        local_x = x.detach().clone().requires_grad_()
        kwargs = dict(
            num_experts=4,
            routing_weights=weights,
            selected_experts=ids,
            hidden_states=local_x,
            independent=mode == "independent",
            ep_group=dist.group.WORLD,
            swiglu_alpha=1.702,
            swiglu_limit=7.0,
            fc1_1_2_weight=m.gate_up_proj.base_layer.weight.chunk(world, 0)[rank].contiguous(),
            fc2_weight=m.down_proj.base_layer.weight.chunk(world, 0)[rank].contiguous(),
        )
        for short, spec in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
            for part in ("A", "B"):
                kwargs[f"lora_{part.lower()}_{short}"] = pairs[f"{spec}.lora_{part}.default.weight"]
            kwargs["lora_scale_" + short] = m._lora_scale_value
        out = _npu_ep_fused_lora_moe_forward(**kwargs)
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.08)
        grads = torch.autograd.grad(out.float().sum(), [local_x] + list(pairs.values()))
        torch.testing.assert_close(grads[0], ref_grads[0], rtol=0.1, atol=0.15)
        for grad, expected in zip(grads[1:], ref_grads[1:], strict=True):
            if mode == "shared":
                grad = grad.contiguous()
                dist.all_reduce(grad)
            else:
                expected = expected.chunk(world, 0)[rank]
            torch.testing.assert_close(grad, expected * world, rtol=0.1, atol=0.3)
        if rank == 0:
            print(f"NPU_EP_PASS {mode}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
