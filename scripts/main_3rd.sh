#!/usr/bin/env bash
# Main-style third-party model launcher for Uni-Agent + ARL rollouts.
set -euo pipefail

SCRIPT_SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${SCRIPT_SRC_ROOT}"
source "${SRC_ROOT}/scripts/load_local_env.sh"
p2a_source_local_env "${SRC_ROOT}"
source "${SRC_ROOT}/scripts/setup.sh"
cd "${SRC_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/main_3rd.sh
  bash scripts/main_3rd.sh --batch config/third_party_batch.example.yaml
  bash scripts/main_3rd.sh --batch .secrets/internal_api_batch.yaml

Model config:
  config/third_party_eval.deepseek.example.yaml

Common overrides:
  P2A_THIRD_PARTY_API_KEY=...
  P2A_THIRD_PARTY_BASE_URL=https://apic1.ohmycdn.com/v1
  P2A_THIRD_PARTY_MODEL=deepseek-v4-flash
  THIRD_PARTY_DATASET=swebench-hard|swebench-verified|r2e-gym-subset
  THIRD_PARTY_DATA_FILE=/path/to/custom.parquet
  P2A_THIRD_PARTY_LIMIT=1
  P2A_THIRD_PARTY_MAX_TURNS=3
  P2A_THIRD_PARTY_PRECOMPUTE_MAPS=1
  P2A_THIRD_PARTY_SYNC_DEPS=0
  P2A_THIRD_PARTY_RUN_TIMEOUT=15m
  P2A_THIRD_PARTY_DB=data/evals/traces.sqlite
  P2A_THIRD_PARTY_EXPERIMENT_ID=third-party-smoke

Outputs:
  data/third_party/<dataset>/<model>/rollouts.jsonl
  data/third_party/<dataset>/<model>/summary.json
  data/third_party/<dataset>/<model>/details.jsonl
  data/third_party/<dataset>/<model>/report.md
  data/evals/traces.sqlite (unless P2A_THIRD_PARTY_DB=0)

Batch outputs:
  storage.db from the batch config, default data/evals/traces.sqlite
  storage.artifacts_dir/<experiment_id>/<stage>/<dataset>/<model>/
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

export PYTHONPATH="${PYTHONPATH:-.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export P2A_DEPLOYMENT="${P2A_DEPLOYMENT:-arl}"
export UNI_AGENT_P2A_TRACE="${UNI_AGENT_P2A_TRACE:-1}"
if [[ "${P2A_DEPLOYMENT,,}" == "arl" ]]; then
  p2a_require_env ARL_GATEWAY_URL
fi

if [[ "${1:-}" == "--batch" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "[third-party] --batch requires a config path" >&2
    usage >&2
    exit 2
  fi
  shift
  exec uv run python -m p2a.third_party_batch --config "$@"
fi

if [[ $# -gt 0 ]]; then
  echo "[third-party] unknown argument: $1" >&2
  usage >&2
  exit 2
fi

THIRD_PARTY_DATASET="${THIRD_PARTY_DATASET:-swebench-hard}"
P2A_THIRD_PARTY_CONFIG="${P2A_THIRD_PARTY_CONFIG:-config/third_party_eval.deepseek.example.yaml}"
P2A_THIRD_PARTY_LIMIT="${P2A_THIRD_PARTY_LIMIT:-1}"
P2A_THIRD_PARTY_OFFSET="${P2A_THIRD_PARTY_OFFSET:-0}"
P2A_THIRD_PARTY_N_PARALLEL="${P2A_THIRD_PARTY_N_PARALLEL:-1}"
P2A_THIRD_PARTY_MAX_TURNS="${P2A_THIRD_PARTY_MAX_TURNS:-3}"
P2A_THIRD_PARTY_MAX_TOKENS="${P2A_THIRD_PARTY_MAX_TOKENS:-1024}"
P2A_THIRD_PARTY_TOOL_INSTALL_TIMEOUT="${P2A_THIRD_PARTY_TOOL_INSTALL_TIMEOUT:-300}"
P2A_THIRD_PARTY_REWARD_TIMEOUT="${P2A_THIRD_PARTY_REWARD_TIMEOUT:-600}"
P2A_THIRD_PARTY_SKIP_TOOL_INSTALL="${P2A_THIRD_PARTY_SKIP_TOOL_INSTALL:-str_replace_editor}"
P2A_THIRD_PARTY_PRECOMPUTE_MAPS="${P2A_THIRD_PARTY_PRECOMPUTE_MAPS:-1}"
P2A_THIRD_PARTY_BONUS_MODE="${P2A_THIRD_PARTY_BONUS_MODE:-dynamic}"
P2A_THIRD_PARTY_BONUS_N_PARALLEL="${P2A_THIRD_PARTY_BONUS_N_PARALLEL:-4}"
P2A_THIRD_PARTY_BONUS_LIMIT="${P2A_THIRD_PARTY_BONUS_LIMIT:-${P2A_THIRD_PARTY_LIMIT}}"
P2A_THIRD_PARTY_BONUS_OFFSET="${P2A_THIRD_PARTY_BONUS_OFFSET:-${P2A_THIRD_PARTY_OFFSET}}"
P2A_THIRD_PARTY_RUN_TIMEOUT="${P2A_THIRD_PARTY_RUN_TIMEOUT:-15m}"
ARTIFACTS_DIR="$(project_artifacts_dir)"
P2A_THIRD_PARTY_DB="${P2A_THIRD_PARTY_DB:-${ARTIFACTS_DIR}/evals/traces.sqlite}"

p2a_setup_select_dataset "${THIRD_PARTY_DATASET}" "${THIRD_PARTY_DATA_FILE:-}"

if [[ "${P2A_THIRD_PARTY_SYNC_DEPS:-0}" == "1" ]]; then
  p2a_setup_sync_deps core
fi

p2a_setup_ensure_dataset "${THIRD_PARTY_DATASET}" "${THIRD_PARTY_DATA_FILE:-}"
DATASET_SLUG="${P2A_SETUP_DATASET_SLUG}"
DATA_FILE="${P2A_SETUP_DATA_FILE}"

MODEL_SLUG="${P2A_THIRD_PARTY_MODEL:-deepseek-v4-flash}"
MODEL_SLUG="${MODEL_SLUG//\//_}"
MODEL_SLUG="${MODEL_SLUG//:/_}"
RUN_DIR="${P2A_THIRD_PARTY_RUN_DIR:-${ARTIFACTS_DIR}/third_party/${DATASET_SLUG}/${MODEL_SLUG}}"
BONUS_MAP_DIR="${P2A_THIRD_PARTY_BONUS_MAP_DIR:-${ARTIFACTS_DIR}/bonus_maps/${DATASET_SLUG}}"
ROLLOUT_OUT="${P2A_THIRD_PARTY_OUT:-${RUN_DIR}/rollouts.jsonl}"
SUMMARY_OUT="${P2A_THIRD_PARTY_SUMMARY_OUT:-${RUN_DIR}/summary.json}"
DETAILS_OUT="${P2A_THIRD_PARTY_DETAILS_OUT:-${RUN_DIR}/details.jsonl}"
REPORT_OUT="${P2A_THIRD_PARTY_REPORT_OUT:-${RUN_DIR}/report.md}"

mkdir -p "${RUN_DIR}"

if [[ "${P2A_THIRD_PARTY_PRECOMPUTE_MAPS}" == "1" ]]; then
  P2A_SETUP_BONUS_MODE="${P2A_THIRD_PARTY_BONUS_MODE}" \
  P2A_SETUP_BONUS_N_PARALLEL="${P2A_THIRD_PARTY_BONUS_N_PARALLEL}" \
  P2A_SETUP_BONUS_LIMIT="${P2A_THIRD_PARTY_BONUS_LIMIT}" \
  P2A_SETUP_BONUS_OFFSET="${P2A_THIRD_PARTY_BONUS_OFFSET}" \
  P2A_SETUP_REBUILD_MAPS="${P2A_THIRD_PARTY_REBUILD_MAPS:-0}" \
    p2a_setup_ensure_bonus_maps "${THIRD_PARTY_DATASET}" "${DATA_FILE}" "${BONUS_MAP_DIR}"
fi

run_cmd=(
  bash scripts/third_party_eval.sh
  --config "${P2A_THIRD_PARTY_CONFIG}"
  --data "${DATA_FILE}"
  --out "${ROLLOUT_OUT}"
  --offset "${P2A_THIRD_PARTY_OFFSET}"
  --n-parallel "${P2A_THIRD_PARTY_N_PARALLEL}"
  --max-turns "${P2A_THIRD_PARTY_MAX_TURNS}"
  --max-tokens "${P2A_THIRD_PARTY_MAX_TOKENS}"
  --tool-install-timeout "${P2A_THIRD_PARTY_TOOL_INSTALL_TIMEOUT}"
  --reward-eval-timeout "${P2A_THIRD_PARTY_REWARD_TIMEOUT}"
  --bonus-map-dir "${BONUS_MAP_DIR}"
  --summary-out "${SUMMARY_OUT}"
  --details-out "${DETAILS_OUT}"
  --report-out "${REPORT_OUT}"
)
if [[ -n "${P2A_THIRD_PARTY_LIMIT}" ]]; then
  run_cmd+=(--limit "${P2A_THIRD_PARTY_LIMIT}")
fi
if [[ -n "${P2A_THIRD_PARTY_DB}" && "${P2A_THIRD_PARTY_DB}" != "0" ]]; then
  run_cmd+=(
    --cache-db "${P2A_THIRD_PARTY_DB}"
    --experiment-id "${P2A_THIRD_PARTY_EXPERIMENT_ID:-third-party-${DATASET_SLUG}}"
    --dataset-name "${DATASET_SLUG}"
    --model-label "${MODEL_SLUG}"
  )
fi
for tool_name in ${P2A_THIRD_PARTY_SKIP_TOOL_INSTALL}; do
  run_cmd+=(--skip-tool-install "${tool_name}")
done
if [[ -n "${P2A_THIRD_PARTY_INSTANCE_IDS:-}" ]]; then
  IFS=',' read -r -a instance_ids <<< "${P2A_THIRD_PARTY_INSTANCE_IDS}"
  for instance_id in "${instance_ids[@]}"; do
    [[ -n "${instance_id}" ]] && run_cmd+=(--instance-id "${instance_id}")
  done
fi

if [[ -n "${P2A_THIRD_PARTY_API_KEY:-}" ]]; then
  API_KEY_STATUS="set"
else
  API_KEY_STATUS="unset"
fi

cat <<EOF
[third-party] dataset: ${DATASET_SLUG}
[third-party] data: ${DATA_FILE}
[third-party] model: ${P2A_THIRD_PARTY_MODEL:-from ${P2A_THIRD_PARTY_CONFIG}}
[third-party] base_url: ${P2A_THIRD_PARTY_BASE_URL:-from ${P2A_THIRD_PARTY_CONFIG}}
[third-party] api_key: ${API_KEY_STATUS}
[third-party] rollouts: ${ROLLOUT_OUT}
[third-party] maps: ${BONUS_MAP_DIR}
EOF

if [[ -n "${P2A_THIRD_PARTY_RUN_TIMEOUT}" && "${P2A_THIRD_PARTY_RUN_TIMEOUT}" != "0" ]]; then
  timeout "${P2A_THIRD_PARTY_RUN_TIMEOUT}" "${run_cmd[@]}"
else
  "${run_cmd[@]}"
fi
