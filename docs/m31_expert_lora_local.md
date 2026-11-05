# M3 expert LoRA on the current Ascend branch

The activation port follows yeyeyeping/VeOmni commit 0188c9ee. This branch
retains its existing NPU operators and supports SwiGLU-OAI in eager and
fused_npu expert LoRA, both shared and independent. The older fused Triton
implementation explicitly rejects SwiGLU-OAI until its backward is ported.

Use external_lib=verl.workers.engine.veomni.m3_lora and
target_modules=all-linear, model.lora.merge=True. Do not also load the old
m3_veomni_lora module. all-linear selects text attention, dense MLP, shared
experts and routed experts; indexer, vision and lm_head are excluded.
model.lora.moe_mode accepts shared or independent (integration default:
independent). Explicit target_parameters may select routed expert layers;
the native wrapper adapts gate, up and down of each selected expert module.

The launcher /mnt/share/l00991686/run_m31_4node_expert_lora.sh defaults to
shared; set MOE_LORA_MODE=independent to opt into per-expert adapters after
memory sizing. SAVE_FREQ defaults to 5. The original launcher is unchanged.
Checkpoint restore requires the same LoRA topology. External adapter import
is still unsupported. Full RL checkpoint resume and multi-node training
must be validated before a long run.

Export collects FSDP tensors, merges gate/up/down separately in FP32 with
one-expert temporary deltas and restores each projection to the base dtype.
It does not modify training weights. Expert tensors remain EP-local before
the existing iter_m3_param reconstructs global expert IDs and public w1/w3/w2
layout. Both shared 2D and independent 3D adapter weights are supported.
Adapter-only vLLM serving is outside this change.

Tests: tests/workers/test_m3_expert_lora.py; set M31_TEST_NPU=1 to enable the
single-device NPU forward/backward tests. Distributed export: torchrun
--standalone --nproc-per-node=2 tests/workers/test_m3_export_ep.py.

## Single-node reduced-model validation

Host: ssh 2241. Model:
`/mnt/share/y00977881/weight/MiniMax-M3.1-Flash-BF16-MoE4-4Layers-1P`.
This is four text layers, three dense MLP layers and one routed MoE layer,
with four routed experts. The single-node launcher defaults to all eight NPUs
(0 through 7), FSDP8/EP4 actor (expert FSDP2) and two independent
TP4/DP1/EP4 rollout replicas. It defaults to independent rank64,
alpha64, all text linear targets plus experts, merged weight sync, three
steps, batch8 with n=2, max response1024 and checkpoint/validation every step.
Rollout uses eager execution for initial validation. This is a correctness
smoke test with shared train/validation data, not a quality benchmark.

Entry: `/mnt/share/l00991686/run_m31_single_expert_lora.sh`.
Tracked copy: `examples/tuning/lora/run_m31_single_npu.sh`.

```
bash /mnt/share/l00991686/run_m31_single_expert_lora.sh check
bash /mnt/share/l00991686/run_m31_single_expert_lora.sh print
# After the selected NPUs and existing Ray instance have been released:
bash /mnt/share/l00991686/run_m31_single_expert_lora.sh head
bash /mnt/share/l00991686/run_m31_single_expert_lora.sh train
```

`check` is the default and only checks files/config. `print` prints argv;
neither starts Ray or training. The head uses a dedicated port (30741) and
refuses to start while another raylet exists. It advertises 64 CPUs by default
(`RAY_NUM_CPUS` override): the eight-worker placement group needs 24 CPUs
in addition to the driver and storage actors; 32 total cannot schedule it. No script stops existing jobs.
To compare shared adapters, use `MOE_LORA_MODE=shared ... train`.
Set `TOTAL_STEPS`, `RESPONSE_LENGTH`, `SAVE_FREQ`, `RUN_DIR` as needed.
ASCEND_RT_VISIBLE_DEVICES accepts four or eight distinct devices and must
be identical for head and train. Four devices select FSDP4/EP4, one rollout
replica and batch4; eight select FSDP8/EP4, two replicas and batch8. EP remains
4 because this checkpoint has only four experts.
Logs, launcher snapshot, source revision/patches and exit_code are saved
under `/mnt/share/l00991686/outputs/m31_single_expert_lora_<timestamp>/`.

## Validation performed

- CPU regression: 57 passed, 2 hardware opt-in cases skipped.
- Single-NPU shared/independent forward and gradient parity: 2 passed.
- Two-NPU EP shared/independent forward and gradient parity: both passed.
- CPU Gloo/DTensor EP2 and EP2 x FSDP2 export: both modes match the
  unsharded public-layout reference exactly.
- VeOmni make quality, touched verl ruff checks, bash syntax and model/shard
  preflight passed. The first four-NPU run completed initial merged vLLM
  refit and 36 validation requests, then exposed missing LoRA kernel binding
  on its first training forward. M3 uses a non-standard OpSlot that skips
  the generic MoE binding, so the bridge now binds its requested LoRA backend
  explicitly. NPU tests exercise the wrapper dispatch instead of calling the
  kernel directly. The post-fix CPU tests report 7 passed, 2 NPU cases
  skipped at the time; after resources were released, both NPU wrapper
  forward/backward tests passed on 2241. Ruff and both launcher topologies
  pass checks. Full training validation remains in progress.
- The standalone distributed scripts are explicit local regression commands;
  they are not added to CI in this local integration.

Logs: `/mnt/share/l00991686/work/m31_expert_lora_20261105/`.

## Checkpoint metadata integration fix

The eight-NPU run `m31_single8_expert_lora_20261105_173116` completed
initial refit/validation and the first actor update. Its checkpoint tensor
shards were saved, but metadata export failed while trying to copy the
VeOmni generated model's `..parallel_plan` import as `.parallel_plan.py`.
The checkpoint config declares only `AutoConfig` in `auto_map`. The FSDP
manager now packages model source only for `AutoModel*` entries; config and
processor `save_pretrained` still save their own custom code normally.
The checkpoint CPU module passes all 20 tests, including config-only and
actual remote-model cases, with and without a model wrapper. The failed
checkpoint is retained as an incomplete artifact; the retry starts fresh.


## LoRA storage precision

New M3 LoRA engine builds call `prepare_m3_lora_dtypes` before FSDP/weight loading:
frozen parameters use BF16, adapters and router weights use FP32, and
`e_score_correction_bias` remains FP32 (buffer or parameter). Routers stay frozen.
Other buffers retain their existing dtype. The bridge enables VeOmni's opt-in
`preserve_frozen_dtype` and requires mixed precision. Router module types join the
existing FSDP mixed-precision exclusion list so their forward also stays FP32;
adapter master weights/optimizer states are FP32 with BF16 mixed-precision compute.
Existing running processes keep their already-loaded implementation; restart is
required for this policy to apply. No checkpoint or active job is modified.


Precision-policy validation (2026-11-05): 60 CPU regression cases passed, two
hardware opt-in cases skipped. The two-rank CPU/Gloo FSDP2 probe passed for both
shared and independent adapters: all six expert adapter tensors had nonzero
FP32 gradients and changed after AdamW; frozen BF16 parameters were unchanged;
router forward and AdamW moments were FP32. Reproduce with
`torchrun --standalone --nproc-per-node=2 tests/workers/test_m3_lora_dtype_fsdp.py`
from the verl checkout. This is a local distributed regression script, not a CI
test. Full NPU training with the new precision policy is still to be validated;
the ongoing main-thread experiment was neither restarted nor modified.
