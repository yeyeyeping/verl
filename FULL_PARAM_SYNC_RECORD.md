# Full-parameter training source synchronization

Captured: 2026-11-12T02:27:19.660717+00:00

Source launcher: `/mnt/share/y00977881/script/vllm_new/multinode_16p_r3/train.sh`.
Default source: `/mnt/share/y00977881/project/veomni_verl_vllm3.1flash_r3/verl`.
Target branch: `3.1flash/full_param`. Previous commit: `8ad8bda884c71e1d2ebefd9d6b8e7875e452bcbb`.

Restore rollout-only AIV mapping and adaptive Geometry3k formatting. Match server checkpoint, MoE parameter replacement and TP/DP setup. Remove desktop-only M3 LoRA integration and its tests/examples.

This commit copies the server implementation; it does not independently fix or
validate its numerical behavior. In particular, CaMem uses pinned CPU backups,
and the server's GBSA metadata uses precision 0. No training or NPU tests run
during synchronization. Server files and running jobs are not modified.

Repository guidance, CI coverage, missing tool links and third-party build
patches are retained. Generated version/build metadata, compiled operators,
caches, logs and backups are not copied. Installed dependencies and compiled
operators are outside this source synchronization. The launcher may override
its default directory through `R3_CODE_ROOT`.

Changed/deleted source and test paths:

- `tests/checkpoint_engine/test_fsdp_lora_only_checkpoint_on_cpu.py`
- `tests/utils/test_vllm_unquant_utils_on_cpu.py`
- `verl/plugin/platform/platform_npu.py`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- `verl/utils/reward_score/geo3k.py`
- `verl/utils/vllm/vllm_unquant_utils.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `docs/m31_expert_lora_local.md`
- `examples/router_replay/MINIMAX_M3_ALIGNMENT.md`
- `examples/tuning/lora/run_m31_single_npu.sh`
- `tests/utils/reward_score/test_geo3k_on_cpu.py`
- `tests/workers/test_m3_expert_lora.py`
- `tests/workers/test_m3_export_ep.py`
- `tests/workers/test_m3_lora_dtype_fsdp.py`
- `tests/workers/test_m3_lora_npu_ep.py`
- `verl/workers/engine/veomni/m3_lora.py`
- `tests/utils/reward_score/test_geo3k_m3.py`

Retained repository-only differences:

- `.claude/skills/issue`
- `.claude/skills/pr`
- `.codex/skills/issue`
- `.codex/skills/pr`

Validation for the coordinated three-repository synchronization:

- All 3,588 selected production files match server Git-blob hashes; absent
  desktop-only runtime modules were removed. Python AST parsing passed.
- Server snapshot was read again before commit and remained unchanged.
- 74 CPU tests passed: attention 46, native MoE-LoRA 16, parameter refit 10,
  Geometry3k 2. Attention also skipped 19 hardware-dependent cases.
- VeOmni full quality commands passed (773 files formatted). Changed verl
  files passed Ruff. Ascend all-file static hooks passed except formatting
  of a copied test; that test was formatted and checked separately, and the
  changed-file long-function check passed.
- Two copied tests have formatting-only changes; production files retain
  the server bytes. Full checkpoint tests could not collect because the
  local CPU environment lacks Ray. NPU and multi-node tests were not run.
