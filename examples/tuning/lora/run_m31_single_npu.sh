#!/usr/bin/env bash
# Single-node M3.1-Flash 4-layer / 4-expert GRPO + expert LoRA validation.
# Target host: ssh 2241 (177.2.24.1 / 172.27.1.104). Never stops existing jobs.
# Default action is read-only check. Explicit head/train are needed to run.
set -eo pipefail
MODE=${1:-check}
if (( $# > 0 )); then shift; fi
ROOT=/mnt/share/l00991686
PROJECT="$ROOT/project/veomni_verl_vllm3.1flash"
ENV_DIR=/mnt/share/y00977881/env/veomni-verl-vllm31flash
export MODEL_PATH=${MODEL_PATH:-/mnt/share/y00977881/weight/MiniMax-M3.1-Flash-BF16-MoE4-4Layers-1P}
export DATA_FILE=${DATA_FILE:-$PROJECT/data/geometry3k-test-verl-balanced36-val0-20261105_004425.parquet}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a NPU_DEVICES <<< "$ASCEND_RT_VISIBLE_DEVICES"
export NPU_COUNT=${#NPU_DEVICES[@]}
export SOCKET_IFNAME=${SOCKET_IFNAME:-data0.172}
LOCAL_IP=$(ip -o -4 addr show dev "$SOCKET_IFNAME" | awk '{split($4,a,"/"); print a[1]; exit}')
[[ -n "$LOCAL_IP" ]] || { echo "No IPv4 on $SOCKET_IFNAME" >&2; exit 2; }
HEAD_IP=$LOCAL_IP
RAY_PORT=${RAY_PORT:-30741}
CACHE_DIR="/tmp/m31-single-lora-$RAY_PORT"
export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH="$CACHE_DIR/ray.token"
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
RESPONSE_LENGTH=${RESPONSE_LENGTH:-1024}
MAX_MODEL_LEN=$((2048 + RESPONSE_LENGTH))
RUN_DIR=${RUN_DIR:-$ROOT/outputs/m31_single_expert_lora_$(date +%Y%m%d_%H%M%S)}

preflight() {
  python3 - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['MODEL_PATH'])
c=json.loads((p/'config.json').read_text())['text_config']
assert c['num_hidden_layers']==4 and c['num_local_experts']==4, 'Expected the specified 4-layer / 4-expert checkpoint'
devices=os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
assert len(devices) in (4, 8) and len(set(devices))==len(devices), 'Select four or eight distinct visible NPUs'
index=json.loads((p/'model.safetensors.index.json').read_text())
for file in set(index['weight_map'].values()):
    assert (p/file).is_file(), f'Missing weight shard: {file}'
for file in ('tokenizer.json','tokenizer_config.json','preprocessor_config.json'):
    assert (p/file).is_file(), f'Missing model asset: {file}'
assert Path(os.environ['DATA_FILE']).is_file(), 'Missing validation data'
print(f"MODEL_OK layers={c['num_hidden_layers']} experts={c['num_local_experts']} moe_layers={sum(c['moe_layer_freq'])}")
print(f'TOPOLOGY single node; {len(devices)} NPUs; train FSDP{len(devices)}/EP4; rollout {len(devices)//4} replicas of TP4/DP1/EP4')
print('VISIBLE_NPUS='+','.join(devices))
PY
}

case "$MODE" in
 check) preflight; exit ;;
 print|head|train) preflight ;;
 *) echo "Usage: bash $0 check|print|head|train [Hydra overrides...]" >&2; exit 2 ;;
esac

source "$ENV_DIR/activate-ascend.sh"
source /mnt/share/y00977881/env/veomni/opp/vendors/m3_base_transformer/bin/set_env.bash
source /mnt/share/y00977881/env/veomni/opp/vendors/m3_exp_transformer/bin/set_env.bash
set -u
export PYTHONPATH="/mnt/share/y00977881/env/veomni/python:$PROJECT/VeOmni:$PROJECT/verl:$PROJECT/vllm:$PROJECT/vllm-ascend-private:${PYTHONPATH:-}"
export VERL_PLATFORM=huawei MODELING_BACKEND=veomni PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
export VLLM_WORKER_MULTIPROC_METHOD=spawn VERL_UPDATE_WEIGHTS_INTERVAL=1
export HCCL_CONNECT_TIMEOUT=300 HCCL_EXEC_TIMEOUT=1800
export HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_SOCKET_IFNAME="$SOCKET_IFNAME" GLOO_SOCKET_IFNAME="$SOCKET_IFNAME" TP_SOCKET_IFNAME="$SOCKET_IFNAME"
export HCCL_IF_IP="$LOCAL_IP" VLLM_HOST_IP="$LOCAL_IP" MASTER_ADDR="$LOCAL_IP"
export PYTHONPYCACHEPREFIX="$CACHE_DIR/pycache" HF_MODULES_CACHE="$CACHE_DIR/hf_modules"
export TRITON_CACHE_DIR="$CACHE_DIR/triton" TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor" VLLM_CACHE_ROOT="$CACHE_DIR/vllm"

if [[ "$MODE" == head ]]; then
    if pgrep -x raylet >/dev/null; then
        echo "An existing Ray instance is running. Refusing to start another; coordinate resource release first." >&2
        exit 2
    fi
    mkdir -p "$CACHE_DIR"
    python - <<'PY'
import os, secrets
from pathlib import Path
p=Path(os.environ['RAY_AUTH_TOKEN_PATH'])
if not p.exists():
    with p.open('x') as f:
        f.write(secrets.token_hex(32))
    p.chmod(0o600)
PY
    exec ray start --head --port="$RAY_PORT" --node-ip-address="$LOCAL_IP" \
        --num-gpus=0 --num-cpus="${RAY_NUM_CPUS:-64}" --object-store-memory=8589934592 \
        "--resources={\"NPU\":$NPU_COUNT}" --include-dashboard=false \
        --dashboard-agent-listen-port=0 --disable-usage-stats
fi

CMD=(python -m verl.trainer.main_ppo
    model_engine=veomni
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "data.train_files=$DATA_FILE"
    "data.val_files=$DATA_FILE"
    "data.train_batch_size=$NPU_COUNT"
    data.max_prompt_length=2048
    "data.max_response_length=$RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=null
    data.truncation=error
    data.dataloader_num_workers=0
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.model.mtp.enable_rollout=False
    actor_rollout_ref.model.lora_rank=64
    actor_rollout_ref.model.lora_alpha=64
    actor_rollout_ref.model.lora.merge=True
    actor_rollout_ref.model.external_lib=verl.workers.engine.veomni.m3_lora
    actor_rollout_ref.model.target_modules=all-linear
    "+actor_rollout_ref.model.lora.moe_mode=${MOE_LORA_MODE:-independent}"
    actor_rollout_ref.model.lora_adapter_path=null
    actor_rollout_ref.nccl_timeout=7200
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE:-5e-5}"
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.ppo_epochs=1
    "actor_rollout_ref.actor.ppo_mini_batch_size=$NPU_COUNT"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_MODEL_LEN"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    "actor_rollout_ref.actor.veomni.fsdp_size=$NPU_COUNT"
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=1
    actor_rollout_ref.actor.veomni.expert_parallel_size=4
    actor_rollout_ref.actor.veomni.moe_implementation=fused_npu
    actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2
    actor_rollout_ref.actor.veomni.minimax_sparse_attention_implementation=npu
    actor_rollout_ref.actor.veomni.enable_full_shard=True
    actor_rollout_ref.actor.veomni.enable_reentrant=False
    actor_rollout_ref.actor.veomni.param_offload=True
    actor_rollout_ref.actor.veomni.optimizer_offload=True
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.expert_parallel_size=4
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.max_num_seqs=4
    actor_rollout_ref.rollout.disable_log_stats=False
    "actor_rollout_ref.rollout.agent.num_workers=$NPU_COUNT"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.n=2
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=128
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=minimax_m3
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:1,video:0}'
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=weights
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=False
    trainer.device=npu
    "trainer.n_gpus_per_node=$NPU_COUNT"
    trainer.nnodes=1
    trainer.logger=console
    trainer.project_name=m31_lora_rl
    trainer.experiment_name=m31_4layer_expert_lora_single
    "trainer.default_local_dir=$RUN_DIR/checkpoints"
    "trainer.rollout_data_dir=$RUN_DIR/rollouts"
    "trainer.validation_data_dir=$RUN_DIR/validation"
    trainer.resume_mode=disable
    trainer.val_before_train=True
    "trainer.save_freq=${SAVE_FREQ:-1}"
    trainer.test_freq=1
    "trainer.total_epochs=${TOTAL_EPOCHS:-100}"
    "trainer.total_training_steps=${TOTAL_STEPS:-3}"
    "++ray_kwargs.ray_init.address=$RAY_ADDRESS"
    "ray_kwargs.ray_init.runtime_env.py_executable=$ENV_DIR/bin/python"
    "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH"

)
CMD+=("$@")

if [[ "$MODE" == print ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
[[ -s "$RAY_AUTH_TOKEN_PATH" ]] || { echo "Start the dedicated head first: bash $0 head" >&2; exit 2; }
mkdir -p "$RUN_DIR"
cp "$0" "$RUN_DIR/launcher.sh"
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
printf '%s\n' "$MODEL_PATH" > "$RUN_DIR/model_path.txt"
for repo in VeOmni verl vllm vllm-ascend-private; do
    git -C "$PROJECT/$repo" rev-parse HEAD > "$RUN_DIR/$repo.commit"
    git -C "$PROJECT/$repo" diff > "$RUN_DIR/$repo.patch"
done
cp "$PROJECT/verl/verl/workers/engine/veomni/m3_lora.py" "$RUN_DIR/m3_lora.py"
echo "RUN_DIR=$RUN_DIR"
cd "$PROJECT/verl"
set +e
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/train.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$RUN_DIR/exit_code"
exit "$rc"
