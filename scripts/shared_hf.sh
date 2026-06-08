#!/usr/bin/env bash
# Shared HuggingFace model defaults for shell launchers.

if [[ -z "${SRC_ROOT:-}" ]]; then
  SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

shared_hf_root() {
  if [[ -n "${P2A_SHARED_ROOT:-}" ]]; then
    mkdir -p "${P2A_SHARED_ROOT}"
    cd "${P2A_SHARED_ROOT}" && pwd
  else
    cd "${SRC_ROOT}/../.." && pwd
  fi
}

shared_models_dir() {
  if [[ -n "${P2A_MODELS_DIR:-}" ]]; then
    mkdir -p "${P2A_MODELS_DIR}"
    cd "${P2A_MODELS_DIR}" && pwd
  else
    local root
    root="$(shared_hf_root)"
    mkdir -p "${root}/models"
    cd "${root}/models" && pwd
  fi
}

default_model_repo() {
  printf '%s\n' "${P2A_MODEL_REPO:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
}

default_model_path() {
  local repo models_dir
  repo="$(default_model_repo)"
  models_dir="$(shared_models_dir)"
  printf '%s/%s\n' "${models_dir}" "${repo##*/}"
}

ensure_model_path() {
  local path repo python_bin
  path="${MODEL_PATH:-$(default_model_path)}"
  repo="$(default_model_repo)"
  python_bin="${P2A_PYTHON:-python3}"

  if [[ -d "${path}" && -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    MODEL_PATH="${path}"
    export MODEL_PATH
    return 0
  fi

  echo "MODEL_PATH=${path} is missing; downloading ${repo} into the shared model cache." >&2
  PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}" "${python_bin}" - "$repo" "$path" <<'PY'
import sys

from p2a.hf_assets import ensure_shared_model

repo, path = sys.argv[1], sys.argv[2]
ensure_shared_model(repo_id=repo, local_dir=path)
PY
  MODEL_PATH="${path}"
  export MODEL_PATH
}
