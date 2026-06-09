#!/usr/bin/env bash
# P2A/ARL training script — adapted from uni-agent/examples/agent_train/single_node_debug.sh
#
# Differences from vanilla Uni-Agent training:
# 1. Uses `python3 -m p2a.main` instead of `python3 -m verl.experimental.fully_async_policy.fully_async_main`
# 2. Passes P2A env vars: P2A_BONUS_MAP_DIR, P2A_M_MAX, P2A_TRACKING_MODE,
#    P2A_EVAL_BONUS_MAP_DIR, P2A_EVAL_NEAR_THRESHOLD, P2A_EVAL_DETAILS_DIR
#
# To run vanilla baseline (no P2A), simply unset P2A_BONUS_MAP_DIR.
#
# Usage:
#   # Baseline (no P2A):
#   bash src/scripts/train_p2a.sh
#
#   # With P2A:
#   P2A_BONUS_MAP_DIR=/path/to/bonus_maps P2A_M_MAX=3.0 bash src/scripts/train_p2a.sh
set -xeuo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SRC_ROOT}/scripts/shared_hf.sh"

UNI_AGENT_DIR="${SRC_ROOT}/uni-agent"
cd "${SRC_ROOT}"

project_name='P2A-SWE-Agent'
exp_name='P2A-GSPO-R2E-Fully-Async'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"$(default_model_path)"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_p2a.train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_hard.parquet"}
RUNTIME_ENV=${RUNTIME_ENV:-"${RAY_DATA_HOME}/data/swe_agent/runtime_env_arl.yaml"}
DEFAULT_AGENT_CONFIG_PATH="${RAY_DATA_HOME}/data/swe_agent/agent_config_arl.yaml"
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-"${DEFAULT_AGENT_CONFIG_PATH}"}

rollout_mode="async"
rollout_name="sglang"

# Uni-Agent's GSPO recipe keeps GRPO advantage estimation and switches the
# actor policy loss to GSPO below.
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=4e-4
clip_ratio_high=4e-4

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 64))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
# Match the Uni-Agent example by default; override with P2A_LOSS_MODE for ablations.
loss_mode=${P2A_LOSS_MODE:-gspo}

temperature=1.0
top_p=1.0
top_k=-1
val_temperature=1.0
val_top_p=0.95
val_top_k=-1

use_dynamic_bsz=True
offload=True
gen_tp=4
train_tp=2
train_pp=1
train_cp=1
train_ep=1
train_etp=1
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

optimizer_offload_fraction=1.0

USE_MBRIDGE=True
USE_DIST_CKPT=False

NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

train_prompt_bsz=0
n_resp_per_prompt=8
train_prompt_mini_bsz=16
total_rollout_steps=200000
test_freq=10
staleness_threshold=1.0
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True

if [[ -n "${P2A_BONUS_MAP_DIR:-}${P2A_EVAL_BONUS_MAP_DIR:-}" && -z "${UNI_AGENT_P2A_TRACE:-}" ]]; then
    export UNI_AGENT_P2A_TRACE=1
fi

ensure_model_path

mkdir -p "$(dirname "${RUNTIME_ENV}")"
if [[ ! -f "${RUNTIME_ENV}" ]]; then
    cp "${UNI_AGENT_DIR}/examples/agent_interaction/runtime_env.yaml" "${RUNTIME_ENV}"
fi
mkdir -p "$(dirname "${AGENT_CONFIG_PATH}")"
if [[ "${AGENT_CONFIG_PATH}" == "${DEFAULT_AGENT_CONFIG_PATH}" || ! -f "${AGENT_CONFIG_PATH}" ]]; then
    cp "${SRC_ROOT}/env/agent_config_arl.yaml" "${AGENT_CONFIG_PATH}"
fi

python3 - "${RUNTIME_ENV}" <<'PY'
import json
import os
import re
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    text = fh.read()

text = re.sub(r'(^\s*PYTHONPATH:\s*).+$', r'\1"uni-agent/verl:uni-agent:."', text, flags=re.MULTILINE)
lines = []
legacy_placeholder_keys = {
    "VEFAAS_FUNCTION_ID",
    "VEFAAS_FUNCTION_ROUTE",
    "VEFAAS_REGION",
    "VOLCE_ACCESS_KEY",
    "VOLCE_SECRET_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
}
legacy_comment_needles = ("if you use vefaas", "if you use modal")
for line in text.splitlines():
    stripped = line.strip()
    if any(needle in stripped.lower() for needle in legacy_comment_needles):
        continue
    key = stripped.split(":", 1)[0]
    if key in legacy_placeholder_keys:
        continue
    lines.append(line)
text = "\n".join(lines) + "\n"

for key in (
    "ARL_GATEWAY_URL",
    "ARL_NAMESPACE",
    "ARL_EXPERIMENT_ID",
    "ARL_TIMEOUT",
    "ARL_STARTUP_TIMEOUT",
    "ARL_MIRROR_REGISTRY",
    "ARL_MIRROR_NAMESPACE",
    "P2A_ARL_IMAGE_OVERRIDES_JSON",
    "P2A_BONUS_MAP_DIR",
    "P2A_M_MAX",
    "P2A_TRACKING_MODE",
    "P2A_EVAL_BONUS_MAP_DIR",
    "P2A_EVAL_NEAR_THRESHOLD",
    "P2A_EVAL_DETAILS_DIR",
    "UNI_AGENT_P2A_TRACE",
):
    value = os.environ.get(key)
    if not value:
        continue
    pattern = rf"(^\s*{re.escape(key)}:\s*).*$"
    replacement = rf"\1{json.dumps(value)}"
    if re.search(pattern, text, flags=re.MULTILINE):
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    elif "env_vars:" in text:
        text = text.rstrip() + f"\n  {key}: {json.dumps(value)}\n"

with open(path, "w", encoding="utf-8") as fh:
    fh.write(text)
PY

# TRAIN_FILE must be the skip-filtered training parquet produced by
# scripts/build_data.py r2e (the *.train.parquet output); bad cases are
# already excluded there, so no separate filter step is needed here.

# --- The one change: use p2a.main instead of verl.experimental.fully_async_policy.fully_async_main ---
ray job submit --no-wait --runtime-env $RUNTIME_ENV \
    -- python3 -m p2a.main \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    hydra.searchpath=[pkg://verl.trainer.config] \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG_PATH} \
    actor_rollout_ref.rollout.agent.default_agent_loop=swe_agent \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.total_epochs=20 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}"
