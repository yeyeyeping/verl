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
with four routed experts. The single-node launcher uses four NPUs (0,1,2,3),
FSDP4/EP4 actor and TP4/DP1/EP4 rollout. It defaults to independent rank64,
alpha64, all text linear targets plus experts, merged weight sync, three
steps, batch4 with n=2, max response1024 and checkpoint/validation every step.
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
refuses to start while another raylet exists. No script stops existing jobs.
To compare shared adapters, use `MOE_LORA_MODE=shared ... train`.
Set `TOTAL_STEPS`, `RESPONSE_LENGTH`, `SAVE_FREQ`, `RUN_DIR` as needed.
Exactly four visible NPUs are required; ASCEND_RT_VISIBLE_DEVICES can select
another set of four, but must be identical for head and train.
Logs, launcher snapshot, source revision/patches and exit_code are saved
under `/mnt/share/l00991686/outputs/m31_single_expert_lora_<timestamp>/`.

## Validation performed

- CPU regression: 57 passed, 2 hardware opt-in cases skipped.
- Single-NPU shared/independent forward and gradient parity: 2 passed.
- Two-NPU EP shared/independent forward and gradient parity: both passed.
- CPU Gloo/DTensor EP2 and EP2 x FSDP2 export: both modes match the
  unsharded public-layout reference exactly.
- VeOmni make quality, touched verl ruff checks, bash syntax and model/shard
  preflight passed. No RL training or actual vLLM refit has been run yet.
- The standalone distributed scripts are explicit local regression commands;
  they are not added to CI in this local integration.

Logs: `/mnt/share/l00991686/work/m31_expert_lora_20261105/`.
