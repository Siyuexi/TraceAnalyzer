#!/usr/bin/env bash
# One-shot baseline launcher. Run from any directory on the Ray head node.
set -euo pipefail

SCRIPT_SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P2A_STAGE_LOCAL_RUNTIME="${P2A_STAGE_LOCAL_RUNTIME:-1}"
SRC_ROOT="${SCRIPT_SRC_ROOT}"
source "${SRC_ROOT}/scripts/load_local_env.sh"
p2a_source_local_env "${SRC_ROOT}"
source "${SRC_ROOT}/scripts/setup.sh"
source "${SRC_ROOT}/scripts/stage_local_runtime.sh"
cd "${SRC_ROOT}"

unset PYTHONPATH PYTHONHOME
unset RAY_ADDRESS
unset P2A_BONUS_MAP_DIR P2A_M_MAX P2A_TRACKING_MODE P2A_CREDIT_GRANULARITY
unset P2A_EVAL_BONUS_MAP_DIR P2A_EVAL_DETAILS_DIR P2A_EVAL_NEAR_THRESHOLD
unset UNI_AGENT_P2A_TRACE
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"

P2A_VENV_DIR="$(p2a_runtime_venv_rel "${SCRIPT_SRC_ROOT}")"
export P2A_VENV_DIR
export UV_PYTHON_INSTALL_DIR="${SCRIPT_SRC_ROOT}/.uv-python"
export UV_PROJECT_ENVIRONMENT="${SCRIPT_SRC_ROOT}/${P2A_VENV_DIR}"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
p2a_source_runtime_profile "${UV_PROJECT_ENVIRONMENT}"
if [[ -z "${P2A_SYNC_DEPS+x}" ]]; then
  P2A_SYNC_DEPS=1
fi

export RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
export RAY_WORKER_HOSTS="${RAY_WORKER_HOSTS:-}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
export RAY_SSH_OPTS="${RAY_SSH_OPTS:--p 36000 -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8}"
p2a_setup_init_data
if [[ -n "${MODEL:-}" ]]; then
  export MODEL="$(resolve_shared_path "${MODEL}")"
else
  export MODEL="$(default_model_path)"
fi
p2a_require_env ARL_GATEWAY_URL
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://localhost:8265}"
export NNODES_TRAIN="${NNODES_TRAIN:-2}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-2}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${RAY_DATA_HOME}/wandb}"

mkdir -p "${RAY_DATA_HOME}" "${DATA}" "${WANDB_DIR}"
DATA="$(cd "${DATA}" && pwd)"
if [[ -d "${MODEL}" ]]; then
  MODEL="$(cd "${MODEL}" && pwd)"
fi
export DATA MODEL

if [[ "${P2A_SYNC_DEPS:-1}" == "1" ]]; then
  p2a_setup_sync_deps train-gpu
fi

p2a_stage_local_runtime "${SCRIPT_SRC_ROOT}"
SRC_ROOT="${P2A_RUNTIME_SRC_ROOT}"
cd "${SRC_ROOT}"
export UV_PYTHON_INSTALL_DIR="${SRC_ROOT}/.uv-python"
export UV_PROJECT_ENVIRONMENT="${SRC_ROOT}/${P2A_VENV_DIR}"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
p2a_source_runtime_profile "${UV_PROJECT_ENVIRONMENT}"

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
RAY_BIN="${UV_PROJECT_ENVIRONMENT}/bin/ray"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
  echo "[baseline] missing ${PYTHON_BIN} or ${RAY_BIN}" >&2
  exit 2
fi

restart_ray_cluster() {
  if [[ "${P2A_RESTART_RAY:-1}" != "1" ]]; then
    echo "[baseline] skipping Ray restart because P2A_RESTART_RAY=${P2A_RESTART_RAY}"
    return
  fi

  local head_ip
  head_ip="${HEAD_IP:-${RAY_HEAD_IP:-${MASTER_IP:-}}}"
  if [[ -z "${head_ip}" ]]; then
    head_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  if [[ -z "${head_ip}" ]]; then
    echo "[baseline] Could not infer HEAD_IP; set HEAD_IP explicitly." >&2
    exit 2
  fi

  echo "[baseline] Restarting Ray cluster from main.sh: head=${head_ip}, workers=${RAY_WORKER_HOSTS}"
  P2A_STAGE_LOCAL_RUNTIME="${P2A_STAGE_LOCAL_RUNTIME}" \
    P2A_SHARED_SRC_ROOT="${P2A_SHARED_SRC_ROOT:-${SCRIPT_SRC_ROOT}}" \
    P2A_LOCAL_ROOT="${P2A_LOCAL_ROOT:-/tmp/p2a-traceanalyzer}" \
    bash "${SCRIPT_SRC_ROOT}/scripts/ray_setup.sh" "${head_ip}" restart-cluster
}

p2a_setup_ensure_dataset r2e-gym-subset
p2a_setup_ensure_dataset swebench-verified
p2a_setup_ensure_dataset swebench-hard

restart_ray_cluster

echo "[baseline] Ray endpoint: ${RAY_API_SERVER_ADDRESS}"
if [[ "${P2A_SHARED_SRC_ROOT:-${SRC_ROOT}}" != "${SRC_ROOT}" ]]; then
  echo "[baseline] shared source: ${P2A_SHARED_SRC_ROOT}"
  echo "[baseline] runtime source: ${SRC_ROOT}"
fi
echo "[baseline] venv: ${UV_PROJECT_ENVIRONMENT}"
echo "[baseline] train: ${DATA}/r2e_gym_subset_p2a.train.parquet"
echo "[baseline] val: ${DATA}/swe_bench_verified_hard.parquet"
echo "[baseline] model: ${MODEL}"

TRAIN_FILE="${DATA}/r2e_gym_subset_p2a.train.parquet" \
TEST_FILE="${DATA}/swe_bench_verified_hard.parquet" \
MODEL_PATH="${MODEL}" \
bash scripts/train_p2a.sh

# Later, for validation graph metrics:
# TEST_FILE="${DATA}/swe_bench_verified_hard.parquet" bash scripts/precompute_eval_bonus_maps.sh
#
# Later, for P2A training, add:
# P2A_BONUS_MAP_DIR=data/bonus_maps/r2e-gym-subset P2A_M_MAX=3.0 P2A_CREDIT_GRANULARITY=step
