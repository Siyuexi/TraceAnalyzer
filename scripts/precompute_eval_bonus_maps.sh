#!/usr/bin/env bash
# Build diagnostic P2A bonus maps for the evaluation parquet.
#
# These maps are not used for training.  They let us score eval rollouts for
# whether the agent reads files/functions on the fault-propagation path.

set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SRC_ROOT}/scripts/shared_hf.sh"
cd "${SRC_ROOT}"

default_data_dir() {
  if [[ -n "${DATA:-}" ]]; then
    local data_dir
    data_dir="$(resolve_shared_path "${DATA}")"
    mkdir -p "${data_dir}"
    cd "${data_dir}" && pwd
  else
    local root
    root="$(shared_hf_root)"
    mkdir -p "${root}/datasets/p2a"
    cd "${root}/datasets/p2a" && pwd
  fi
}

DATA_DIR="$(default_data_dir)"
EVAL_FILE="${EVAL_FILE:-${TEST_FILE:-${DATA_DIR}/swe_bench_verified_hard.parquet}}"
P2A_EVAL_BONUS_MAP_DIR="${P2A_EVAL_BONUS_MAP_DIR:-${DATA_DIR}/eval_bonus_maps}"
P2A_EVAL_BONUS_MODE="${P2A_EVAL_BONUS_MODE:-dynamic}"
P2A_EVAL_BONUS_N_PARALLEL="${P2A_EVAL_BONUS_N_PARALLEL:-16}"
P2A_EVAL_BONUS_SKIP_EXISTING="${P2A_EVAL_BONUS_SKIP_EXISTING:-1}"
P2A_DEPLOYMENT="${P2A_DEPLOYMENT:-arl}"
PYTHONPATH=".:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess:${PYTHONPATH:-}"
export P2A_DEPLOYMENT PYTHONPATH

if [[ ! -f "${EVAL_FILE}" ]]; then
  cat >&2 <<EOF
Evaluation parquet not found: ${EVAL_FILE}

Build it first, for example:
  PYTHONPATH=.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess \\
    uv run python scripts/build_data.py swebench-hard --out "${EVAL_FILE}"
EOF
  exit 2
fi

mkdir -p "${P2A_EVAL_BONUS_MAP_DIR}"

cmd=(
  uv run python p2a/precompute/precompute_bonus_maps.py
  "${EVAL_FILE}"
  --output_dir "${P2A_EVAL_BONUS_MAP_DIR}"
  --mode "${P2A_EVAL_BONUS_MODE}"
  --n_parallel "${P2A_EVAL_BONUS_N_PARALLEL}"
)

if [[ "${P2A_EVAL_BONUS_SKIP_EXISTING}" == "0" ]]; then
  cmd+=(--rebuild)
fi
if [[ -n "${P2A_EVAL_BONUS_LIMIT:-}" ]]; then
  cmd+=(--limit "${P2A_EVAL_BONUS_LIMIT}")
fi
if [[ -n "${P2A_EVAL_BONUS_OFFSET:-}" ]]; then
  cmd+=(--offset "${P2A_EVAL_BONUS_OFFSET}")
fi
if [[ "${P2A_EVAL_BONUS_SAVE_SIDECARS:-0}" != "0" ]]; then
  cmd+=(--save_trace_sidecars)
  if [[ -n "${P2A_EVAL_TRACE_SIDECAR_DIR:-}" ]]; then
    cmd+=(--trace_sidecar_dir "${P2A_EVAL_TRACE_SIDECAR_DIR}")
  fi
fi

printf 'Eval parquet: %s\n' "${EVAL_FILE}"
printf 'Eval bonus maps: %s\n' "${P2A_EVAL_BONUS_MAP_DIR}"
printf 'Deployment: %s\n' "${P2A_DEPLOYMENT}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
