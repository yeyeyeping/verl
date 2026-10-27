# MiniMax M3 RL on Ascend

The VeOmni engine keeps each actor/reference mesh and scopes initialization,
forward/backward, clipping, checkpointing, and weight iteration to that mesh.
This uses the installed VeOmni `init_parallel_state_from_config` and
`use_parallel_state` APIs. No generated modeling files are modified.

## Weight synchronization

`minimax_m3_vl` exports the public checkpoint layout, including separate
`w1/w3/w2` expert weights, dense/shared gate-up splitting, indexer names,
vision encoder names, and both projector MLPs. Shapes and layer/expert counts
come from the model config. EP-local stacks are reconstructed across the EP
group after FSDP gathering. Already-global stacks are not broadcast again.
Unsupported names/shapes fail explicitly. Delta/shard export is not supported
for this public-layout conversion; use full synchronization.

NPU rollout uses host shared memory. Tensors larger than the configured bucket
(including the M3 embedding and LM head) stream through that same buffer in
chunks; increasing the bucket above the largest tensor is unnecessary.
Optimizer device-to-host copies are synchronous.

`VERL_UPDATE_WEIGHTS_INTERVAL=1` preserves the normal every-step update.
A positive value greater than one skips intervening updates, while the first
update, updates after adding replicas, and calls with no step always synchronize.
Skipped updates wake/resume rollout and retain the previous weight version.
Skipping is currently restricted to NPU vLLM, whose level-1 sleep preserves
weights. A larger interval deliberately
uses an older rollout policy and changes training behavior.

## Prompt alignment

HF M3 expands each image into vision-start, image patches, and vision-end.
Before vLLM submission, this entire block becomes one image placeholder.
After vLLM preprocessing, expanded prompt IDs must equal the original training
IDs exactly. An image-count, boundary, preprocessing, or position mismatch
raises instead of silently yielding incomparable logprobs.

M3 uses ordinary 1D text positions; it is not assigned Qwen's `get_rope_index`.
Expanded video prompts currently fail explicitly: reversing per-frame
timestamps requires the original video metadata, which the image collapse
helper does not reconstruct.

## Validation scope

CPU regressions cover token boundaries/counts, shape failures, expert ordering,
actor/reference state restoration, mandatory initial sync, replica
wake-up on skipped updates, and multi-chunk shared-memory transfer. Actual
multi-node HCCL/vLLM startup and end-to-end RL numerical agreement still need a
training run with the intended launcher, model, and dataset.
