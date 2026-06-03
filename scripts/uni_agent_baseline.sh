#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: src/scripts/uni_agent_baseline.sh prepare|data|debug|all

prepare  Copy Uni-Agent veFaaS runtime/agent configs into $RAY_DATA_HOME.
data     Generate R2E-Gym-Subset train parquet and SWE-Bench Verified eval parquet.
debug    Run Uni-Agent single-node debug launcher.
all      Run prepare, data, then debug.
EOF
}

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VEFAAS_ENV="${SRC_DIR}/.secrets/vefaas_env.sh"
if [[ -f "${DEFAULT_VEFAAS_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${DEFAULT_VEFAAS_ENV}"
fi

UNI_AGENT_DIR="${SRC_DIR}/uni-agent"
RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
DATA_DIR="${RAY_DATA_HOME}/data/swe_agent"
RUNTIME_ENV="${RUNTIME_ENV:-${DATA_DIR}/runtime_env.yaml}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-${DATA_DIR}/agent_config.yaml}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/r2e_gym_subset_filtered.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/swe_bench_verified_vefaas.parquet}"

require_uni_agent() {
  if [[ ! -d "${UNI_AGENT_DIR}/uni_agent" || ! -d "${UNI_AGENT_DIR}/verl" ]]; then
    echo "Missing Uni-Agent or nested verl under ${UNI_AGENT_DIR}" >&2
    echo "Run: git -C src submodule update --init --recursive" >&2
    exit 1
  fi
}

write_runtime_env_from_shell() {
  python3 - "$RUNTIME_ENV" <<'PY'
import json
import os
import re
import sys

path = sys.argv[1]
keys = [
    "VEFAAS_FUNCTION_ID",
    "VEFAAS_FUNCTION_ROUTE",
    "VOLCE_ACCESS_KEY",
    "VOLCE_SECRET_KEY",
    "VEFAAS_REGION",
    "UNI_AGENT_P2A_TRACE",
]

with open(path, "r", encoding="utf-8") as fh:
    text = fh.read()

env_vars_seen = "env_vars:" in text
for key in keys:
    value = os.environ.get(key)
    if not value:
        continue
    if key == "VEFAAS_FUNCTION_ROUTE":
        value = value.rstrip("/")
    pattern = rf"(^\s*{re.escape(key)}:\s*).*$"
    replacement = rf"\1{json.dumps(value)}"
    if re.search(pattern, text, flags=re.MULTILINE):
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    elif env_vars_seen:
        text = text.rstrip() + f"\n  {key}: {json.dumps(value)}\n"

with open(path, "w", encoding="utf-8") as fh:
    fh.write(text)
PY
}

prepare() {
  require_uni_agent
  mkdir -p "$DATA_DIR"

  if [[ ! -f "$RUNTIME_ENV" ]]; then
    cp "${UNI_AGENT_DIR}/examples/agent_interaction/runtime_env.yaml" "$RUNTIME_ENV"
  fi
  if [[ ! -f "$AGENT_CONFIG_PATH" ]]; then
    cp "${UNI_AGENT_DIR}/examples/agent_interaction/agent_config_vefaas.yaml" "$AGENT_CONFIG_PATH"
  fi

  write_runtime_env_from_shell

  local debug_concurrency="${UNI_AGENT_DEBUG_CONCURRENCY:-4}"
  local debug_max_turns="${UNI_AGENT_DEBUG_MAX_TURNS:-10}"
  local debug_action_timeout="${UNI_AGENT_DEBUG_ACTION_TIMEOUT:-120}"
  local debug_eval_timeout="${UNI_AGENT_DEBUG_EVAL_TIMEOUT:-300}"

  sed -i -E "s/^  concurrency: .*/  concurrency: ${debug_concurrency}/" "$AGENT_CONFIG_PATH"
  sed -i -E "s/^    max_turns: .*/    max_turns: ${debug_max_turns}/" "$AGENT_CONFIG_PATH"
  sed -i -E "s/^    action_timeout: .*/    action_timeout: ${debug_action_timeout}/" "$AGENT_CONFIG_PATH"
  sed -i -E "s/^    eval_timeout: .*/    eval_timeout: ${debug_eval_timeout}/" "$AGENT_CONFIG_PATH"

  cat <<EOF
Prepared Uni-Agent baseline config:
  RUNTIME_ENV=${RUNTIME_ENV}
  AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH}
  TRAIN_FILE=${TRAIN_FILE}
  TEST_FILE=${TEST_FILE}

If ${DEFAULT_VEFAAS_ENV} exists, veFaaS secrets are loaded from it automatically.
Check runtime_env.yaml before launching only if that file is absent or intentionally bypassed.
EOF
}

data() {
  require_uni_agent
  mkdir -p "$DATA_DIR"
  cd "$UNI_AGENT_DIR"
  DEPLOYMENT=vefaas python examples/data_preprocess/r2e_gym_subset_filtered.py --local-save-dir "$DATA_DIR"
  DEPLOYMENT=vefaas python examples/data_preprocess/swe_bench_verified.py --local-save-dir "$DATA_DIR"
}

debug() {
  require_uni_agent
  if [[ ! -f "$RUNTIME_ENV" ]]; then
    echo "Missing $RUNTIME_ENV; run prepare first." >&2
    exit 1
  fi
  if [[ ! -f "$AGENT_CONFIG_PATH" ]]; then
    echo "Missing $AGENT_CONFIG_PATH; run prepare first." >&2
    exit 1
  fi
  if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "MODEL_PATH is not set; set it to the Qwen checkpoint path." >&2
    exit 1
  fi

  cd "$UNI_AGENT_DIR"
  export RAY_DATA_HOME
  export TRAIN_FILE
  export TEST_FILE
  export RUNTIME_ENV
  export AGENT_CONFIG_PATH
  bash examples/agent_train/single_node_debug.sh
}

cmd="${1:-}"
case "$cmd" in
  prepare)
    prepare
    ;;
  data)
    data
    ;;
  debug)
    debug
    ;;
  all)
    prepare
    data
    debug
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
