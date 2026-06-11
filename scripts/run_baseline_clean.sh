#!/usr/bin/env bash
# Clean one-shot baseline launcher. Run from any directory on the Ray head node
# after scripts/ray_setup.sh has already been run on every GPU node.
set -euo pipefail

SCRIPT_SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_SRC_ROOT}/scripts/stage_local_runtime.sh"
p2a_stage_local_runtime "${SCRIPT_SRC_ROOT}"
SRC_ROOT="${P2A_RUNTIME_SRC_ROOT}"
cd "${SRC_ROOT}"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" ]]; then
  echo "[baseline] uv not found on PATH" >&2
  exit 2
fi

unset PYTHONPATH PYTHONHOME
unset RAY_ADDRESS
unset P2A_BONUS_MAP_DIR P2A_M_MAX P2A_TRACKING_MODE
unset P2A_EVAL_BONUS_MAP_DIR P2A_EVAL_DETAILS_DIR P2A_EVAL_NEAR_THRESHOLD
unset UNI_AGENT_P2A_TRACE
if [[ "${P2A_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi

export UV_PYTHON_INSTALL_DIR="${SRC_ROOT}/.uv-python"
export UV_PROJECT_ENVIRONMENT="${SRC_ROOT}/.venv"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

export RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
export DATA="${DATA:-${SRC_ROOT}/../../datasets/p2a}"
export MODEL="${MODEL:-${SRC_ROOT}/../../models/Qwen3-Coder-30B-A3B-Instruct}"
export ARL_GATEWAY_URL="${ARL_GATEWAY_URL:-http://118.145.210.10:8080}"
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8265}"
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
  if [[ "${P2A_REBUILD_VENV:-0}" == "1" || ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    "${UV_BIN}" python install --managed-python 3.11
    "$("${UV_BIN}" python find --managed-python --no-project 3.11)" -m venv --clear --copies "${UV_PROJECT_ENVIRONMENT}"
  fi
  UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" "${UV_BIN}" sync --locked --extra train --extra gpu
fi

PYTHON_BIN="${UV_PROJECT_ENVIRONMENT}/bin/python"
RAY_BIN="${UV_PROJECT_ENVIRONMENT}/bin/ray"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
  echo "[baseline] missing ${PYTHON_BIN} or ${RAY_BIN}" >&2
  exit 2
fi

build_data() {
  local target="$1"
  shift
  if [[ -f "${target}" ]]; then
    echo "[baseline] data exists: ${target}"
    return
  fi
  PYTHONPATH=.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess \
    "${PYTHON_BIN}" scripts/build_data.py "$@"
}

build_data "${DATA}/r2e_gym_subset_p2a.train.parquet" \
  r2e --out "${DATA}/r2e_gym_subset_p2a.parquet"
build_data "${DATA}/swe_bench_verified.parquet" \
  swebench-verified --out "${DATA}/swe_bench_verified.parquet"
build_data "${DATA}/swe_bench_verified_hard.parquet" \
  swebench-hard --out "${DATA}/swe_bench_verified_hard.parquet"

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
# TEST_FILE="${DATA}/swe_bench_verified_hard.parquet" \
#   P2A_EVAL_BONUS_MAP_DIR="${DATA}/eval_bonus_maps" bash scripts/precompute_eval_bonus_maps.sh
#
# Later, for P2A training, add:
# P2A_BONUS_MAP_DIR=../../p2a/bonus_maps P2A_M_MAX=3.0
