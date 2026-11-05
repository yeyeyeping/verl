"""Two-process Gloo check of EP-local expert LoRA merge and global export.

Run with torchrun --standalone --nproc-per-node=2 tests/workers/test_m3_export_ep.py.
"""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from test_m3_expert_lora import ToyM3
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

from verl.workers.engine.veomni.m3_lora import MergedState, merged_m3_lora_weights, wrap_m3_lora
from verl.workers.engine.veomni.m3_weight_export import iter_m3_param


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world in (2, 4)
    fsdp = world // 2
    ep_rank = rank // fsdp
    groups = [dist.new_group([i, i + fsdp]) for i in range(fsdp)]
    ps = SimpleNamespace(ep_enabled=True, ep_rank=ep_rank, ep_size=2, ep_group=groups[rank % fsdp])
    # Four ranks exercise EP2 x FSDP2; two ranks exercise EP2 alone.
    meshes = [DeviceMesh("cpu", list(range(e * fsdp, (e + 1) * fsdp))) for e in range(2)]
    for mode in ("shared", "independent"):
        torch.manual_seed(99)
        model = wrap_m3_lora(ToyM3(), 4, 7, "all-linear", moe_mode=mode)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if ".lora_B." in name:
                    p.normal_(0, 0.1)
        state = model.state_dict()
        expected = {
            n: t.clone()
            for n, t in iter_m3_param(
                merged_m3_lora_weights(model, state, "cpu"), model.config, SimpleNamespace(ep_enabled=False), "cpu"
            )
        }
        local = {}
        for name, tensor in state.items():
            if ".experts." in name and tensor.ndim == 3:
                tensor = tensor.chunk(2, 0)[ep_rank].clone()
            if tensor.ndim > 0:
                tensor = distribute_tensor(tensor, meshes[ep_rank], [Shard(1 if tensor.ndim == 3 else 0)])
            local[name] = tensor
        template = merged_m3_lora_weights(model, state, "cpu")
        merged = MergedState(local, template.scales, "cpu", template.experts)
        actual = {n: t.clone() for n, t in iter_m3_param(merged, model.config, ps, "cpu")}
        assert set(actual) == set(expected)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        if rank == 0:
            print(f"EP_EXPORT_PASS mode={mode} tensors={len(actual)}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
