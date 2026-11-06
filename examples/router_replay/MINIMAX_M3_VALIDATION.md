# MiniMax M3 VL R3 integration

Integration date: 2026-11-06.

## Companion repositories

Use the `3.1flash/router_replay` branch in all three repositories:

| Repository | Base commit | Change |
| --- | --- | --- |
| VeOmni | `bce66007fda5597bd51f833a1f2310c5ccccafe4` | GPU/NPU router hook and regenerated modeling |
| verl | `463f8de7fd6f75f51e4ad3dcd85dc4f6545b26b0` | Rollout decoder slots to sparse FFN slots |
| vllm-ascend-private | `b14f4748fc3bbf26882b4936ff816112c69fb193` | Single-DP TP route capture reconstruction |

VeOmni replays expert IDs before sigmoid weight gathering and normalization.
The actor computes weights and gradients from its own router logits. verl
selects sparse layers using `text_config.mlp_layer_types`; R2 recordings
already contain only MoE layers and are not projected again. Installing a
controller on another model resets its layer mapping.

Ascend reconstructs the full DP-local routing sequence before writing the
capture buffer. MC2/FUSED_MC2 use global tail padding; ALLTOALL follows
potentially uneven `tensor_split` chunks. The new single-DP sharded path
rejects sequence-parallel `replace_allreduce=True` rather than guessing its
layout. Existing multi-DP capture behavior is unchanged.

Enable:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
# Pass these overrides to the verl launcher:
actor_rollout_ref.actor.veomni.router_replay.mode=R3
actor_rollout_ref.rollout.enable_rollout_routing_replay=True
```

Propagate the environment variable to rollout workers. Restart workers after
updating the Python source. This feature changes no native Ascend operators,
so it does not itself require recompiling C++/CANN extensions. A non-editable
installation still needs the updated Python package installed.

## Validation scope

The functional integration excludes temporary JSONL routing dumps, hidden-state
hooks, fixed-token diagnostic launchers and FP32 precision substitutions.
Existing opt-in replay shape checks remain part of the base controller.

Regression coverage includes native router equality with replay disabled,
forced expert IDs and gradients, GPU/NPU generated hook wiring, mixed
dense/sparse mapping, wrapped/text/multimodal model configs, controller reuse,
R2 layout preservation, invalid layouts, recompute/duplicate handling, and
Ascend TP capture with small/uneven/padded token counts.


Companion feature commits verified with this integration:

- VeOmni: [`0cae50f061a1`](https://github.com/yeyeyeping/VeOmni/commit/0cae50f061a1f4979f1e554ee9c6c09774efe554).
- vllm-ascend-private: [`15806d24c14d`](https://github.com/yeyeyeping/vllm-ascend-private/commit/15806d24c14d5a15ce55d9a323da1d2633856fee).

The verl commit is the commit containing this record.

## Integration checks (2026-11-06)

The final desktop sources were compared with the tested snapshot by SHA-256
after normalizing CRLF to LF. All 16 tracked R3 files matched. The extra
validation document is the seventeenth file in the combined change.

| Repository | Command / check | Result |
| --- | --- | --- |
| VeOmni | `pytest tests/models/test_minimax_m3_vl_ops.py -k router_replay_preserves_native -q` | 2 passed |
| VeOmni | `pytest tests/utils/test_moe_router_replay.py -q` | 23 passed |
| verl | `pytest tests/utils/veomni/test_router_replay_on_cpu.py tests/workers/test_router_replay_engine_helpers_on_cpu.py -q` | 38 passed |
| vllm-ascend-private | `pytest tests/ut/patch/worker/test_patch_routed_experts_capture.py -q` | 29 passed |
| VeOmni | GPU/NPU MiniMax `check_config` with Transformers 5.16.1 | Both passed |
| VeOmni | `make quality` with repository-pinned Ruff 0.13.1 | Passed, 773 files formatted |
| vllm-ascend-private | `bash format.sh ci` on isolated WSL snapshot | Passed all hooks |
| All three | Changed-file Ruff checks and `git diff --check` | Passed |

Total: 92 passing regression tests. Generated AST differences are restricted
to `MiniMaxM3VLTopKRouter` in both backends. The capture tests mock collectives;
this integration did not launch a new distributed training or NPU performance
run. Earlier runtime evidence is listed separately below.

Validation reused installed dependencies without changing the main training
environment. Remote code generation/tests used an isolated directory; a pinned
Ruff wheel was installed only there because the shared tool version differed
and its package-download proxy was unavailable. WSL format checks used one
worker (`PRE_COMMIT_NO_CONCURRENCY=1`, `RAYON_NUM_THREADS=1`) after a high
concurrency attempt encountered memory pressure. Shell line endings were
normalized only in the disposable WSL snapshot so tracked Windows CRLF shell
scripts could execute. Existing CLAUDE.md and skill-symlink workspace changes
were preserved and excluded from all feature commits.

## Earlier runtime evidence

These are historical measurements from the isolated R3 investigation, not a
new training run of this Git integration. The source report was
`outputs/m31_router_replay/REPORT.txt`. Configuration: 60 layers (3 dense,
57 MoE), 8 experts/top-4, BF16, one node with 8 Ascend950DT devices, actor
FSDP2/EP8, rollout TP8/EP enabled, DP1, sequence parallelism disabled,
router LoRA disabled and initialized LoRA B=0.

On the same frozen production-rollout batch (32 samples, 3,840 scored tokens),
using FP32 actor logprob normalization:

| Mode | Mean absolute logprob difference | P99 absolute difference | Absolute difference > 0.1 |
| --- | --- | --- | --- |
| Replay disabled | 0.186810363 | 1.298592958 | 1,915 / 3,840 (49.869792%) |
| R3 | 0.081513858 | 0.566692208 | 973 / 3,840 (25.338542%) |

All 889,656 valid token-layer routing rows matched, including all 218,880
scored-response rows. Repeated actor forwards were identical. Missing final
input-token routes legitimately fall back to native routing and must not be
counted as valid replay coverage.

R3 reduces routing divergence but does not establish identical training and
inference numerics. The earlier one-step run had zero advantage/gradient;
nonzero-LoRA equivalence, repeated-merge safety, convergence, and the proposed
5–10-step learning-rate comparison are not validated by these results.
