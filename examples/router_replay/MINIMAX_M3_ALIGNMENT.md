# MiniMax M3 alignment integration

The synchronized branches are verl `3.1flash/lora_verl`, VeOmni
`3.1flash/lora_expert`, and vllm-ascend `3.1flash/lora_ascend`. They include
the earlier `3.1flash/full_param` and `3.1flash/router_replay` commits. The branch name does not select full-weight
training: LoRA remains opt-in and its forward mode defaults to `split`.

## Activation

Load these VeOmni, verl and vllm-ascend sources together. For R3, set
`VLLM_USE_V2_MODEL_RUNNER=1` in both the launcher and Ray worker runtime environment,
and pass:

```text
actor_rollout_ref.actor.veomni.router_replay.mode=R3
actor_rollout_ref.rollout.enable_rollout_routing_replay=True
```

For the tested independent-expert LoRA mode, additionally pass:

```text
actor_rollout_ref.model.external_lib=verl.workers.engine.veomni.m3_lora
actor_rollout_ref.model.lora.merge=True
+actor_rollout_ref.model.lora.moe_mode=independent
+actor_rollout_ref.model.lora.forward_mode=merged
actor_rollout_ref.actor.veomni.moe_implementation=fused_npu
actor_rollout_ref.actor.veomni.low_precision_reduce_scatter_comm=False
```

Merged forward requires FP32 adapter owners, BF16 frozen base weights and zero
LoRA dropout. It preserves gradients and does not mutate the base. Its temporary
matrix materialization increases memory use. `lora.merge=True` controls export;
it is not a substitute for `forward_mode=merged`.

## Numerical fixes

- Fused main/index Q/K Norm rounds `1 + weight` to the weight storage dtype
  before FP32 multiplication, matching the existing unfused path. The 512-token
  fusion eligibility limit and the unfused path are unchanged.
- GBSA metadata and backward use `softmax_precision=0`, matching the installed CANN gradient API;
  explicit packed used lengths remain intact. Forward `inner_precise=0` is a
  different operator parameter and is unchanged.
- Geo3k already accepts matching `<mm:think>` tags, with 0.1 format reward.
- No `norm_row_fp32` experimental override or long-run diagnostic hooks are added.

## Evidence and limits

Transferred sources match the 2026-11-07 validated migration package: 92 R3
CPU regressions, 22 LoRA CPU regressions, and six NPU norm/graph regressions passed
in that earlier remote validation. The fixed-input NPU norm comparison used two
in-place weight updates and 1/64/512/513-token inputs; mean output error dropped
from about 1e-3 to 0--1e-6. These are norm-output errors, not full-model logprobs.

The earlier desktop synchronization changed GBSA test expectations from
precision 0 to 1; the current snapshot restores the supported precision 0. Local static checks and source
hash comparisons are separate from the earlier remote runtime results. No new
NPU training, multi-node launch, or server configuration change is part of this
desktop synchronization.

## Running snapshot update (2026-11-08)

The current four-node run enables
`actor_rollout_ref.actor.veomni.enable_pageable_activation_offload=True`.
This option defaults to false. It wraps the training forward outside the
checkpoint boundary with VeOmni `custom_save_on_cpu(gpu_limit_in_gb=0.0,
pin_memory=False)`; forward-only evaluation is unchanged. Pair it with the
VeOmni frozen-weight transpose exclusion to avoid restoring a duplicate full
vocabulary head. Merged LoRA still requires zero LoRA dropout.

The earlier precision-one experiment described above was reverted in the live
source: the installed GBSA gradient metadata operator accepts precision zero.
Use precision zero for both metadata and backward, as in the synchronized
VeOmni implementation.
