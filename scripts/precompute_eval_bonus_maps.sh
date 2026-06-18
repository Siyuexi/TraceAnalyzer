#!/usr/bin/env bash
# Build diagnostic P2A bonus maps for the evaluation parquet.
#
# These maps are not used for training.  They let us score eval rollouts for
# whether the agent reads files/functions on the fault-propagation path.

set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SRC_ROOT}/scripts/setup.sh"
cd "${SRC_ROOT}"

DATA_DIR="$(p2a_setup_data_dir)"
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

printf 'Eval parquet: %s\n' "${EVAL_FILE}"
printf 'Eval bonus maps: %s\n' "${P2A_EVAL_BONUS_MAP_DIR}"
printf 'Deployment: %s\n' "${P2A_DEPLOYMENT}"

P2A_SETUP_BONUS_MODE="${P2A_EVAL_BONUS_MODE}" \
P2A_SETUP_BONUS_N_PARALLEL="${P2A_EVAL_BONUS_N_PARALLEL}" \
P2A_SETUP_BONUS_LIMIT="${P2A_EVAL_BONUS_LIMIT:-}" \
P2A_SETUP_BONUS_OFFSET="${P2A_EVAL_BONUS_OFFSET:-}" \
P2A_SETUP_REBUILD_MAPS="$([[ "${P2A_EVAL_BONUS_SKIP_EXISTING}" == "0" ]] && printf '1' || printf '0')" \
P2A_SETUP_SAVE_TRACE_SIDECARS="${P2A_EVAL_BONUS_SAVE_SIDECARS:-0}" \
P2A_SETUP_TRACE_SIDECAR_DIR="${P2A_EVAL_TRACE_SIDECAR_DIR:-}" \
  p2a_setup_ensure_bonus_maps swebench-hard "${EVAL_FILE}" "${P2A_EVAL_BONUS_MAP_DIR}"
