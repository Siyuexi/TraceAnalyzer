#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: src/scripts/uni_agent_arl.sh prepare|data|smoke|debug|all

prepare  Write ARL runtime_env.yaml and agent_config.yaml into $RAY_DATA_HOME.
data     Generate ARL-backed R2E-Gym-Subset train parquet.
smoke    Boot one ARL sandbox and verify SDK runtime persistence/upload.
debug    Run Uni-Agent single-node debug launcher using ARL config.
all      Run prepare, data, then debug.
EOF
}

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNI_AGENT_DIR="${SRC_DIR}/uni-agent"
RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
DATA_DIR="${RAY_DATA_HOME}/data/swe_agent"
RUNTIME_ENV="${RUNTIME_ENV:-${DATA_DIR}/runtime_env_arl.yaml}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-${DATA_DIR}/agent_config_arl.yaml}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/r2e_gym_subset_filtered.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/r2e_gym_subset_filtered.parquet}"

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
    "ARL_GATEWAY_URL",
    "ARL_NAMESPACE",
    "ARL_EXPERIMENT_ID",
    "ARL_TIMEOUT",
    "ARL_STARTUP_TIMEOUT",
    "ARL_MIRROR_REGISTRY",
    "ARL_MIRROR_NAMESPACE",
    "P2A_ARL_ENTERPRISE_REPOS",
    "P2A_ARL_IMAGE_OVERRIDES_JSON",
    "P2A_ARL_DISABLE_MIRROR",
    "UNI_AGENT_P2A_TRACE",
    "P2A_BONUS_MAP_DIR",
    "P2A_M_MAX",
    "P2A_TRACKING_MODE",
]

with open(path, "r", encoding="utf-8") as fh:
    text = fh.read()

text = re.sub(r'(^\s*PYTHONPATH:\s*).+$', r'\1"uni-agent/verl:uni-agent:."', text, flags=re.MULTILINE)
env_vars_seen = "env_vars:" in text
for key in keys:
    value = os.environ.get(key)
    if not value:
        continue
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
  cp "${SRC_DIR}/env/agent_config_arl.yaml" "$AGENT_CONFIG_PATH"
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
Prepared Uni-Agent ARL config:
  RUNTIME_ENV=${RUNTIME_ENV}
  AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH}
  TRAIN_FILE=${TRAIN_FILE}
  TEST_FILE=${TEST_FILE}
EOF
}

data() {
  require_uni_agent
  mkdir -p "$DATA_DIR"
  # Reuse uni-agent's stock R2E data prep verbatim (submodule unmodified). The
  # `deployment.type: arl` is supplied by agent_config_arl.yaml and deep-merged
  # in at agent-loop init; the parquet only carries per-instance image +
  # post_setup_cmd + reward, exactly uni-agent's shape.
  cd "$UNI_AGENT_DIR"
  DEPLOYMENT=vefaas python examples/data_preprocess/r2e_gym_subset_filtered.py --local-save-dir "$DATA_DIR"
}

smoke() {
  require_uni_agent
  cd "$SRC_DIR"
  local image="${ARL_SMOKE_IMAGE:-enterprise-public-cn-beijing.cr.volces.com/r2e-gym-subset/a95245e37f:latest}"
  python -m env.smoke --image "$image"
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

  cd "$SRC_DIR"
  export RAY_DATA_HOME
  export TRAIN_FILE
  export TEST_FILE
  export RUNTIME_ENV
  export AGENT_CONFIG_PATH
  bash uni-agent/examples/agent_train/single_node_debug.sh
}

cmd="${1:-}"
case "$cmd" in
  prepare)
    prepare
    ;;
  data)
    data
    ;;
  smoke)
    smoke
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
