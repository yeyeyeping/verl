"""Opt-in M3 text and expert LoRA integration for the local verl/VeOmni stack.

Loaded via actor_rollout_ref.model.external_lib. Native VeOmni wraps/loads
adapters before FSDP2/EP. Rollout receives full, per-tensor merged weights;
the training base is never merged in place. No installed source is changed.
"""

import logging
import re
from contextlib import contextmanager
from datetime import timedelta

import torch
from torch.distributed.tensor import DTensor
from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
from veomni.lora.layers import LoraLinear
from veomni.lora.moe_layers import LoraIndependentExperts, LoraSharedExperts

EXPERT_TYPES = (LoraIndependentExperts, LoraSharedExperts)

LOG = logging.getLogger(__name__)


@contextmanager
def isolated_weight_metadata():
    """Keep loader metadata off the default group's mixed CPU/NPU channel.

    The original loader uses default-group object collectives interleaved with
    parameter distribution. Use a dedicated CPU Gloo group for those objects;
    parameter tensors and every other distributed operation retain their group.
    This changes only module_utils' namespace during the initial weight load.
    """
    from veomni.models import module_utils

    dist = module_utils.dist
    group = dist.new_group(backend="gloo", timeout=timedelta(minutes=10))

    class LoaderDistributed:
        def __getattr__(self, name):
            return getattr(dist, name)

        def broadcast_object_list(self, object_list, src=None, group=None, device=None, group_src=None):
            if group is None:
                group = metadata_group
                device = torch.device("cpu")
            return dist.broadcast_object_list(object_list, src=src, group=group, device=device, group_src=group_src)

    metadata_group = group
    module_utils.dist = LoaderDistributed()
    LOG.warning(
        "M3_WEIGHT_METADATA dedicated CPU Gloo group rank=%d size=%d", dist.get_rank(group), dist.get_world_size(group)
    )
    try:
        yield
    finally:
        module_utils.dist = dist
        dist.destroy_process_group(group)


def wrap_m3_lora(
    model, rank, alpha, target_modules=None, target_parameters=None, moe_mode="independent", moe_implementation="eager"
):
    """Configure M3 text LoRA; all-linear includes dense, shared and routed experts."""
    if getattr(model.config, "model_type", None) != "minimax_m3_vl":
        raise ValueError("M3 LoRA is restricted to minimax_m3_vl")
    all_linear = target_modules == "all-linear"
    allowed = re.compile(
        r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        r"|mlp\.(shared_experts\.)?(gate_up_proj|down_proj))"
    )
    requested = target_modules if target_modules is not None else ["q_proj", "k_proj", "v_proj", "o_proj"]
    if not all_linear and isinstance(requested, str):
        raise ValueError("M3 target_modules must be all-linear or a list of text projections")
    if not all_linear and any("index" in name for name in requested):
        raise ValueError("M3 indexer LoRA is unsupported")
    targets = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear)
        and allowed.fullmatch(name)
        and (all_linear or any(name == t or name.endswith("." + t) for t in requested))
    ]
    if not targets and not target_parameters:
        raise ValueError("No M3 LoRA targets matched")
    if not all_linear:
        missing = [t for t in requested if not any(n == t or n.endswith("." + t) for n in targets)]
        if missing:
            raise ValueError(f"Unsupported or unmatched M3 LoRA targets: {missing}")
    expert_targets = list(target_parameters or [])
    if all_linear:
        expert_targets = [f"model.language_model.layers.*.mlp.experts.{p}" for p in ("gate_up_proj", "down_proj")]
    for name in expert_targets:
        if not re.fullmatch(r"model\.language_model\.layers\.(\d+|\*)\.mlp\.experts\.(gate_up_proj|down_proj)", name):
            raise ValueError(f"Unsupported M3 expert target: {name}")
    cfg = VeOmniLoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        target_parameters=expert_targets or None,
        moe_mode=moe_mode if expert_targets else None,
        lora_dropout=0.0,
        bias="none",
    )
    wrapped = VeOmniLoraModel(model, cfg)
    trainable = [name for name, p in wrapped.named_parameters() if p.requires_grad]
    if not trainable or any(".lora_A." not in name and ".lora_B." not in name for name in trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")
    expert_count = sum(isinstance(m, EXPERT_TYPES) for m in wrapped.modules())
    if expert_targets and not expert_count:
        raise RuntimeError("Expert LoRA requested but no experts were wrapped")
    if expert_count:
        if moe_implementation not in ("eager", "fused_npu"):
            raise ValueError("M3 expert LoRA supports only eager or fused_npu")
        # M3's non-standard OpSlot skips the generic MoE patch that normally
        # binds these pointers. Bind explicitly before the first EP forward.
        from veomni.lora.ops import bind_lora_moe_kernels

        bind_lora_moe_kernels(moe_implementation.removeprefix("fused_"))
        LOG.warning("M3_LORA expert kernel backend=%s", moe_implementation)
    LOG.warning(
        "M3_LORA dense_targets=%d expert_layers=%d mode=%s rank=%d alpha=%d trainable=%d",
        len(targets),
        expert_count,
        moe_mode,
        rank,
        alpha,
        sum(p.numel() for p in wrapped.parameters() if p.requires_grad),
    )
    return wrapped


@contextmanager
def adapters_disabled(module):
    # Native LoraLinear returns the base result for an absent adapter name.
    layers = [(m, m.active_adapter) for m in module.modules() if isinstance(m, LoraLinear)]
    experts = [
        (m, m._lora_scale_value, m.lora_scaling.detach().clone())
        for m in module.modules()
        if isinstance(m, EXPERT_TYPES)
    ]
    if not layers and not experts:
        raise RuntimeError("Reference requested from LoRA actor with no adapters")
    try:
        for m, _ in layers:
            m.active_adapter = "__m3_reference_disabled__"
        for m, _, _ in experts:
            m._lora_scale_value = 0.0
            m.lora_scaling.zero_()
        yield
    finally:
        for m, scale, buffer in experts:
            m._lora_scale_value = scale
            m.lora_scaling.copy_(buffer)
        for m, adapter in layers:
            m.active_adapter = adapter


def _full(tensor, device):
    tensor = tensor.detach().to(device)
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


class MergedState:
    """Merge one projection at a time without modifying training state.

    DTensor.full_tensor reconstructs the FSDP shard. Expert tensors remain
    EP-local when the training mesh is EP-local; iter_m3_param owns EP exchange.
    FP32 deltas are bounded to one expert, not the whole expert stack.
    """

    def __init__(self, state, scales, device, experts=None):
        self.state, self.scales, self.device = state, scales, device
        self.experts = experts or {}

    def _pair(self, root, adapter):
        return tuple(_full(self.state[f"{root}.lora_{part}.{adapter}.weight"], self.device) for part in ("A", "B"))

    def _expert_projection(self, root, projection, value):
        scale, adapter = self.experts[root]
        base = _full(value, self.device)
        if base.ndim != 3:
            raise ValueError(f"Expected 3D expert base: {root}.{projection}")
        specs = ("gate_proj", "up_proj") if projection == "gate_up_proj" else ("down_proj",)
        pairs = [self._pair(root + "." + spec, adapter) for spec in specs]
        result = torch.empty_like(base)
        for expert in range(base.shape[0]):
            halves = base[expert].chunk(len(specs), dim=0)
            outputs = result[expert].chunk(len(specs), dim=0)
            for weight, out, (a, b) in zip(halves, outputs, pairs, strict=True):
                if a.ndim == 3 and b.ndim == 3:
                    if a.shape[0] != base.shape[0] or b.shape[0] != base.shape[0]:
                        raise ValueError(f"Expert LoRA/base EP extent mismatch: {root}")
                    ae, be = a[expert], b[expert]
                elif a.ndim == 2 and b.ndim == 2:
                    ae, be = a, b
                else:
                    raise ValueError(f"Invalid expert LoRA layout: {root}")
                if (be.shape[0], ae.shape[1]) != tuple(weight.shape) or be.shape[1] != ae.shape[0]:
                    raise ValueError(f"Expert LoRA projection shape mismatch: {root}")
                out.copy_((weight.float() + (be.float() @ ae.float()) * scale).to(base.dtype))
        return result

    @torch.no_grad()
    def items(self):
        for name, value in self.state.items():
            if ".lora_A." in name or ".lora_B." in name or name.endswith(".lora_scaling"):
                continue
            clean = name.removeprefix("base_model.model.")
            if ".base_layer." in name:
                root, suffix = name.rsplit(".base_layer.", 1)
                parent, projection = root.rsplit(".", 1)
                if parent in self.experts:
                    if suffix != "weight" or projection not in ("gate_up_proj", "down_proj"):
                        raise ValueError(f"Unexpected expert base parameter: {name}")
                    value = self._expert_projection(parent, projection, value)
                    clean = root.removeprefix("base_model.model.")
                else:
                    clean = clean.replace(".base_layer.", ".")
                    if suffix == "weight":
                        if root not in self.scales:
                            raise ValueError(f"No adapter scaling for {root}")
                        base = _full(value, self.device)
                        a, b = self._pair(root, "default")
                        value = (base.float() + (b.float() @ a.float()) * self.scales[root]).to(base.dtype)
            yield clean, value


def merged_m3_lora_weights(module, state, device):
    scales = {name: m.scaling["default"] for name, m in module.named_modules() if isinstance(m, LoraLinear)}
    experts = {
        name: (m._lora_scale_value, m.adapter_name) for name, m in module.named_modules() if isinstance(m, EXPERT_TYPES)
    }
    if not scales and not experts:
        raise RuntimeError("No LoRA layers found during rollout export")
    return MergedState(state, scales, device, experts)


def install():
    from verl.workers.engine.veomni import transformer_impl as impl

    cls = impl.VeOmniEngine
    if getattr(cls, "_m3_expert_lora_bridge", False):
        return
    if getattr(cls, "_m3_native_lora_bridge", False):
        raise RuntimeError("Remove external_lib=m3_veomni_lora before loading M3 expert LoRA")
    original_build = cls._build_model_optimizer
    original_export = cls.get_per_tensor_param
    original_disable = cls.disable_adapter

    def build(self):
        if not self._is_lora:
            return original_build(self)
        if self.model_config.lora_adapter_path:
            raise ValueError("Use verl checkpoint resume for this bridge; adapter import is unsupported")
        if not self.model_config.lora.get("merge", False):
            raise ValueError("M3 LoRA requires model.lora.merge=True for ordinary vLLM weight sync")
        ops = impl._build_ops_implementation_config(self.engine_config)
        mp = impl.MixedPrecisionConfig(enable=self.engine_config.mixed_precision)
        module = impl.build_foundation_model(
            config_path=self._get_model_config_path(),
            weights_path=self.model_config.local_path,
            torch_dtype="float32" if mp.enable else "bfloat16",
            attn_implementation=self.engine_config.attn_implementation,
            ops_implementation=ops,
            init_device=self.engine_config.init_device,
        )
        module = wrap_m3_lora(
            module,
            self.model_config.lora_rank,
            self.model_config.lora_alpha,
            self.model_config.target_modules,
            self.model_config.target_parameters,
            self.model_config.lora.get("moe_mode", "independent"),
            moe_implementation=self.engine_config.moe_implementation,
        )
        self._maybe_apply_async_activation_offload(module)
        with isolated_weight_metadata():
            module = impl.build_parallelize_model(
                module,
                init_device=self.engine_config.init_device,
                weights_path=self.model_config.local_path,
                is_peft_model=True,
                enable_reshard_after_forward=self.engine_config.enable_full_shard,
                mixed_precision=mp,
                low_precision_reduce_scatter_comm=self.engine_config.low_precision_reduce_scatter_comm,
                enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing,
                enable_fsdp_offload=self.engine_config.enable_fsdp_offload,
                basic_modules=list(
                    set(getattr(module, "_no_split_modules", None) or []) | set(self.engine_config.basic_modules)
                ),
                enable_reentrant=self.engine_config.enable_reentrant,
                enable_forward_prefetch=self.engine_config.forward_prefetch,
                broadcast_model_weights_from_rank0=True,
                fqn_to_index_mapping=impl.load_safetensors_index(self.model_config.local_path),
            )
        self.module = module
        if self.engine_config.forward_only:
            self.optimizer = self.lr_scheduler = None
        else:
            self.optimizer = self._build_optimizer(module)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

    @impl._with_engine_parallel_state
    def export(self, **kwargs):
        if not self._is_lora:
            return original_export(self, **kwargs)
        from verl.workers.engine.veomni.m3_weight_export import iter_m3_param

        manual = not getattr(self, "_uses_fsdp2_cpu_offload_policy", False)
        if manual:
            impl.load_veomni_model_to_gpu(self.module)
        state = self.module.state_dict()
        merged = merged_m3_lora_weights(self.module, state, impl.get_device_id())
        if self._is_offload_param and manual:
            impl.offload_veomni_model_to_cpu(self.module)

        @torch.no_grad()
        def generate():
            yield from iter_m3_param(merged, self.module.config, self.parallel_state, impl.get_device_id())

        return impl._iter_with_engine_parallel_state(self, generate()), None

    def disable(self):
        return adapters_disabled(self.module) if self._is_lora else original_disable(self)

    cls._build_model_optimizer = build
    cls.get_per_tensor_param = export
    cls.disable_adapter = disable
    cls._m3_expert_lora_bridge = True


install()
