#!/usr/bin/env bash
# Idempotent setup helpers for data, dependencies, and eval bonus maps.

if [[ -z "${SRC_ROOT:-}" ]]; then
  SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
source "${SRC_ROOT}/scripts/lib.sh"

p2a_setup_usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup.sh deps <core|dev|train-gpu>
  bash scripts/setup.sh data <swebench-hard|swebench-verified|swebench-pro|r2e-gym-subset>
  bash scripts/setup.sh maps <swebench-hard|swebench-verified|swebench-pro|r2e-gym-subset>
  bash scripts/setup.sh train
  bash scripts/setup.sh third-party

The setup steps are idempotent: existing parquets, venvs, and map files are reused
unless the corresponding rebuild environment variable is set. Third-party setup
does not sync dependencies by default; set P2A_SETUP_DEPS_PROFILE=core when you
want an explicit CPU/core sync in a disposable environment.
EOF
}

p2a_setup_data_dir() {
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

p2a_setup_init_data() {
  DATA="$(p2a_setup_data_dir)"
  export DATA
}

p2a_setup_select_dataset() {
  local dataset custom_file
  dataset="${1:-swebench-hard}"
  custom_file="${2:-}"
  p2a_setup_init_data

  P2A_SETUP_DATASET_SLUG=""
  P2A_SETUP_DATA_FILE=""
  P2A_SETUP_BUILD_CMD=()
  case "${dataset}" in
    swebench-hard|swe-bench-hard|hard)
      P2A_SETUP_DATASET_SLUG="swebench-hard"
      P2A_SETUP_DATA_FILE="${custom_file:-${DATA}/swe_bench_verified_hard.parquet}"
      P2A_SETUP_BUILD_CMD=(swebench-hard --out "${P2A_SETUP_DATA_FILE}")
      ;;
    swebench-verified|swe-bench-verified|verified)
      P2A_SETUP_DATASET_SLUG="swebench-verified"
      P2A_SETUP_DATA_FILE="${custom_file:-${DATA}/swe_bench_verified.parquet}"
      P2A_SETUP_BUILD_CMD=(swebench-verified --out "${P2A_SETUP_DATA_FILE}")
      ;;
    swebench-pro|swe-bench-pro|swebenchpro|swe-pro|pro)
      P2A_SETUP_DATASET_SLUG="swebench-pro"
      P2A_SETUP_DATA_FILE="${custom_file:-${DATA}/swe_bench_pro.parquet}"
      P2A_SETUP_BUILD_CMD=(swebench-pro --out "${P2A_SETUP_DATA_FILE}")
      if [[ -n "${P2A_SWEBENCH_PRO_SCRIPTS_DIR:-}" ]]; then
        P2A_SETUP_BUILD_CMD+=(--scripts-dir "${P2A_SWEBENCH_PRO_SCRIPTS_DIR}")
      elif [[ -n "${SWEBENCH_PRO_SCRIPTS_DIR:-}" ]]; then
        P2A_SETUP_BUILD_CMD+=(--scripts-dir "${SWEBENCH_PRO_SCRIPTS_DIR}")
      fi
      ;;
    r2e|r2e-gym|r2e-gym-subset)
      P2A_SETUP_DATASET_SLUG="r2e-gym-subset"
      P2A_SETUP_DATA_FILE="${custom_file:-${DATA}/r2e_gym_subset_p2a.train.parquet}"
      P2A_SETUP_BUILD_CMD=(r2e --out "${DATA}/r2e_gym_subset_p2a.parquet" --train-out "${P2A_SETUP_DATA_FILE}")
      ;;
    *)
      echo "[setup] unknown dataset: ${dataset}" >&2
      echo "[setup] expected one of: swebench-hard, swebench-verified, swebench-pro, r2e-gym-subset" >&2
      return 2
      ;;
  esac
  export P2A_SETUP_DATASET_SLUG P2A_SETUP_DATA_FILE
}

p2a_setup_python_cmd() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
  elif [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
    printf '%s\n' "${UV_PROJECT_ENVIRONMENT}/bin/python"
  else
    printf '%s\n' "uv run python"
  fi
}

p2a_setup_ensure_dataset() {
  local dataset custom_file python_cmd
  dataset="${1:-swebench-hard}"
  custom_file="${2:-}"
  p2a_setup_select_dataset "${dataset}" "${custom_file}"
  if [[ -f "${P2A_SETUP_DATA_FILE}" ]]; then
    if [[ "${P2A_SETUP_DATASET_SLUG}" == "swebench-pro" ]]; then
      python_cmd="$(p2a_setup_python_cmd)"
      PYTHONPATH=".:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess:${PYTHONPATH:-}" \
        ${python_cmd} scripts/build_data.py validate-swebench-pro --path "${P2A_SETUP_DATA_FILE}"
    fi
    echo "[setup] data exists: ${P2A_SETUP_DATA_FILE}"
    return 0
  fi
  if [[ "${P2A_SETUP_BUILD_DATA:-1}" != "1" ]]; then
    echo "[setup] data file missing and P2A_SETUP_BUILD_DATA=0: ${P2A_SETUP_DATA_FILE}" >&2
    return 2
  fi

  echo "[setup] building data: ${P2A_SETUP_DATA_FILE}"
  python_cmd="$(p2a_setup_python_cmd)"
  PYTHONPATH=".:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess:${PYTHONPATH:-}" \
    ${python_cmd} scripts/build_data.py "${P2A_SETUP_BUILD_CMD[@]}"
}

p2a_setup_sync_deps() {
  local profile uv_bin
  profile="${1:-core}"
  uv_bin="${UV_BIN:-$(command -v uv || true)}"
  if [[ -z "${uv_bin}" ]]; then
    echo "[setup] uv not found on PATH" >&2
    return 2
  fi

  case "${profile}" in
    none|skip)
      echo "[setup] dependency sync skipped"
      ;;
    core)
      "${uv_bin}" sync --locked
      ;;
    dev)
      "${uv_bin}" sync --locked --extra dev
      ;;
    train-gpu)
      if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
        echo "[setup] UV_PROJECT_ENVIRONMENT is required for train-gpu deps" >&2
        return 2
      fi
      if [[ "${P2A_REBUILD_VENV:-0}" == "1" || ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
        "${uv_bin}" python install --managed-python 3.11
        "$("${uv_bin}" python find --managed-python --no-project 3.11)" -m venv --clear --copies "${UV_PROJECT_ENVIRONMENT}"
      fi
      UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT}" "${uv_bin}" sync --locked --extra train --extra gpu
      ;;
    *)
      echo "[setup] unknown dependency profile: ${profile}" >&2
      return 2
      ;;
  esac
}

p2a_setup_ensure_bonus_maps() {
  local dataset custom_file output_dir mode n_parallel limit offset artifacts_dir
  dataset="${1:-swebench-hard}"
  custom_file="${2:-}"
  output_dir="${3:-}"
  p2a_setup_ensure_dataset "${dataset}" "${custom_file}"

  artifacts_dir="$(project_artifacts_dir)"
  output_dir="${output_dir:-${artifacts_dir}/bonus_maps/${P2A_SETUP_DATASET_SLUG}}"
  mode="${P2A_SETUP_BONUS_MODE:-dynamic}"
  n_parallel="${P2A_SETUP_BONUS_N_PARALLEL:-16}"
  limit="${P2A_SETUP_BONUS_LIMIT:-}"
  offset="${P2A_SETUP_BONUS_OFFSET:-}"
  mkdir -p "${output_dir}"

  local cmd=(
    uv run python p2a/precompute/precompute_bonus_maps.py
    "${P2A_SETUP_DATA_FILE}"
    --output_dir "${output_dir}"
    --mode "${mode}"
    --n_parallel "${n_parallel}"
  )
  if [[ "${P2A_SETUP_REBUILD_MAPS:-0}" == "1" ]]; then
    cmd+=(--rebuild)
  fi
  if [[ -n "${limit}" && "${limit}" != "all" ]]; then
    cmd+=(--limit "${limit}")
  fi
  if [[ -n "${offset}" && "${offset}" != "0" ]]; then
    cmd+=(--offset "${offset}")
  fi
  if [[ "${P2A_SETUP_SAVE_TRACE_SIDECARS:-0}" != "0" ]]; then
    cmd+=(--save_trace_sidecars)
    if [[ -n "${P2A_SETUP_TRACE_SIDECAR_DIR:-}" ]]; then
      cmd+=(--trace_sidecar_dir "${P2A_SETUP_TRACE_SIDECAR_DIR}")
    fi
  fi

  P2A_SETUP_BONUS_MAP_DIR="${output_dir}"
  export P2A_SETUP_BONUS_MAP_DIR
  echo "[setup] bonus maps: ${P2A_SETUP_BONUS_MAP_DIR}"
  P2A_DEPLOYMENT="${P2A_DEPLOYMENT:-arl}" \
  PYTHONPATH=".:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess:${PYTHONPATH:-}" \
    "${cmd[@]}"
}

p2a_setup_train() {
  p2a_setup_sync_deps "${P2A_SETUP_DEPS_PROFILE:-train-gpu}"
  p2a_setup_ensure_dataset r2e-gym-subset
  p2a_setup_ensure_dataset swebench-verified
  p2a_setup_ensure_dataset swebench-hard
}

p2a_setup_third_party() {
  p2a_setup_select_dataset "${THIRD_PARTY_DATASET:-swebench-hard}" "${THIRD_PARTY_DATA_FILE:-}"
  p2a_setup_sync_deps "${P2A_SETUP_DEPS_PROFILE:-none}"
  p2a_setup_ensure_dataset "${THIRD_PARTY_DATASET:-swebench-hard}" "${THIRD_PARTY_DATA_FILE:-}"
  if [[ "${P2A_SETUP_PRECOMPUTE_MAPS:-1}" == "1" ]]; then
    p2a_setup_ensure_bonus_maps "${THIRD_PARTY_DATASET:-swebench-hard}" "${THIRD_PARTY_DATA_FILE:-}" "${P2A_SETUP_BONUS_MAP_DIR:-}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "${SRC_ROOT}"
  cmd="${1:-help}"
  shift || true
  case "${cmd}" in
    deps)
      p2a_setup_sync_deps "${1:-core}"
      ;;
    data)
      p2a_setup_ensure_dataset "${1:-swebench-hard}" "${2:-}"
      printf '%s\n' "${P2A_SETUP_DATA_FILE}"
      ;;
    maps)
      p2a_setup_ensure_bonus_maps "${1:-swebench-hard}" "${2:-}" "${3:-}"
      printf '%s\n' "${P2A_SETUP_BONUS_MAP_DIR}"
      ;;
    train)
      p2a_setup_train
      ;;
    third-party|third_party|3rd)
      p2a_setup_third_party
      ;;
    -h|--help|help)
      p2a_setup_usage
      ;;
    *)
      p2a_setup_usage >&2
      exit 2
      ;;
  esac
fi
