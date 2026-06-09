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
SRC_ROOT="${SRC_DIR}"
source "${SRC_DIR}/scripts/shared_hf.sh"
UNI_AGENT_DIR="${SRC_DIR}/uni-agent"
RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
DATA_DIR="${RAY_DATA_HOME}/data/swe_agent"
RUNTIME_ENV="${RUNTIME_ENV:-${DATA_DIR}/runtime_env_arl.yaml}"
AGENT_CONFIG_PATH="${AGENT_CONFIG_PATH:-${DATA_DIR}/agent_config_arl.yaml}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/r2e_gym_subset_p2a.train.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/r2e_gym_subset_p2a.train.parquet}"

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
    "P2A_ARL_IMAGE_OVERRIDES_JSON",
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
  # Build via the canonical self-contained builder: it writes the full parquet plus a
  # skip-filtered *.train.parquet (bad_instances excluded), with pair-diag image refs.
  # deployment.type: arl is supplied by agent_config_arl.yaml and deep-merged at
  # agent-loop init; the parquet carries per-instance image + post_setup_cmd + reward.
  cd "$SRC_DIR"
  PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
    uv run python scripts/build_data.py r2e --out "${DATA_DIR}/r2e_gym_subset_p2a.parquet"
}

smoke() {
  require_uni_agent
  cd "$SRC_DIR"
  # Default to a real pair-diag image taken from the built R2E parquet (build_data.py
  # writes pair-diag refs); otherwise require ARL_SMOKE_IMAGE. Never default to enterprise.
  local image="${ARL_SMOKE_IMAGE:-}"
  if [[ -z "$image" && -f "$TRAIN_FILE" ]]; then
    image="$(uv run python -c "
import json, pandas as pd
ei = pd.read_parquet('$TRAIN_FILE')['extra_info'].iloc[0]
ei = json.loads(ei) if isinstance(ei, str) else ei
print(ei['tools_kwargs']['env']['deployment']['image'])
" 2>/dev/null || true)"
  fi
  if [[ -z "$image" ]]; then
    echo "Set ARL_SMOKE_IMAGE to a pair-diag image, or run '${BASH_SOURCE[0]} data' first to build ${TRAIN_FILE}." >&2
    exit 1
  fi
  uv run python -m env.smoke --image "$image"
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
  ensure_model_path

  cd "$SRC_DIR"

  # TRAIN_FILE (and an R2E TEST_FILE) must be the skip-filtered training parquet
  # produced by scripts/build_data.py r2e (*.train.parquet) — bad cases are
  # already excluded there, so no separate filter step is needed here.

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
