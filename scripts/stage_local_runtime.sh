#!/usr/bin/env bash
# Stage the shared checkout and Python runtime onto node-local disk.

p2a_stage_enabled() {
  [[ "${P2A_STAGE_LOCAL_RUNTIME:-${P2A_LOCAL_RUNTIME:-0}}" == "1" ]]
}

p2a_preferred_venv_rel() {
  local source_root="$1"
  if [[ -n "${P2A_VENV_DIR:-}" ]]; then
    printf '%s\n' "${P2A_VENV_DIR%/}"
  elif [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    if [[ "${UV_PROJECT_ENVIRONMENT}" == /* ]]; then
      printf '%s\n' "${UV_PROJECT_ENVIRONMENT#${source_root}/}"
    else
      printf '%s\n' "${UV_PROJECT_ENVIRONMENT%/}"
    fi
  elif [[ -x "${source_root}/.venv-cu128/bin/python" ]]; then
    printf '.venv-cu128\n'
  else
    printf '.venv\n'
  fi
}

p2a_runtime_venv_rel() {
  local source_root="$1"
  local rel
  source_root="$(cd "${source_root}" && pwd -P)"
  rel="$(p2a_preferred_venv_rel "${source_root}")"
  rel="${rel#./}"
  rel="${rel%/}"
  case "${rel}" in
    ""|/*|../*|*/../*|*/..)
      echo "[stage] P2A_VENV_DIR/UV_PROJECT_ENVIRONMENT must point inside ${source_root}: ${rel}" >&2
      return 2
      ;;
  esac
  # With P2A_VENV_DIR set, an inherited UV_PROJECT_ENVIRONMENT may legitimately
  # point at a parent launcher's staged runtime; the explicit name wins.
  if [[ -z "${P2A_VENV_DIR:-}" && -n "${UV_PROJECT_ENVIRONMENT:-}" && "${UV_PROJECT_ENVIRONMENT}" == /* && "${UV_PROJECT_ENVIRONMENT}" != "${source_root}/${rel}" ]]; then
    echo "[stage] UV_PROJECT_ENVIRONMENT must point inside ${source_root}: ${UV_PROJECT_ENVIRONMENT}" >&2
    return 2
  fi
  printf '%s\n' "${rel}"
}

p2a_abs_dir() {
  local path="$1"
  mkdir -p "${path}"
  (cd "${path}" && pwd -P)
}

p2a_source_runtime_profile() {
  local venv_path="${1:-${UV_PROJECT_ENVIRONMENT:-}}"
  if [[ -n "${venv_path}" && -f "${venv_path}/p2a-cu128.env" ]]; then
    # shellcheck disable=SC1091
    source "${venv_path}/p2a-cu128.env"
  fi
}

p2a_runtime_stamp() {
  local source_root="$1"
  local venv_rel="${2:-}"
  if [[ -z "${venv_rel}" ]]; then
    venv_rel="$(p2a_runtime_venv_rel "${source_root}")"
  fi
  (
    cd "${source_root}"
    printf 'source=%s\n' "${source_root}"
    printf 'venv=%s\n' "${venv_rel}"
    git rev-parse HEAD 2>/dev/null || true
    git submodule status --recursive 2>/dev/null || true
    for path in pyproject.toml uv.lock "${venv_rel}/pyvenv.cfg" "${venv_rel}/bin/python" "${venv_rel}/bin/ray" .uv-python; do
      if [[ -e "${path}" ]]; then
        stat -c '%n %s %Y' "${path}" 2>/dev/null || ls -l "${path}"
      fi
    done
  ) | sha256sum | awk '{print $1}'
}

p2a_rewrite_staged_paths() {
  local old_root="$1"
  local new_root="$2"
  local venv_rel="${3:-.venv}"
  local patch_python
  local roots=("${old_root}")
  patch_python="$(command -v python3 || command -v python || true)"
  if [[ -z "${patch_python}" ]]; then
    echo "[stage] python3 or python is required to rewrite staged venv paths." >&2
    return 2
  fi

  if [[ -f "${new_root}/${venv_rel}/bin/ray" ]]; then
    local shebang
    IFS= read -r shebang < "${new_root}/${venv_rel}/bin/ray" || true
    shebang="${shebang#\#!}"
    if [[ "${shebang}" == */"${venv_rel}"/bin/python* ]]; then
      roots+=("${shebang%%/${venv_rel}/bin/python*}")
    fi
  fi
  if [[ -f "${new_root}/${venv_rel}/pyvenv.cfg" ]]; then
    local line root
    while IFS= read -r line; do
      if [[ "${line}" == *"/.uv-python/"* ]]; then
        root="${line#*= }"
        roots+=("${root%%/.uv-python/*}")
      fi
    done < "${new_root}/${venv_rel}/pyvenv.cfg"
  fi
  if [[ -L "${new_root}/${venv_rel}" || -L "${new_root}/.uv-python" ]]; then
    echo "[stage] staged ${venv_rel}/.uv-python must be real local directories, not symlinks." >&2
    return 2
  fi

  "${patch_python}" - "${new_root}" "${venv_rel}" "${roots[@]}" <<'PY'
import sys
from pathlib import Path

new = sys.argv[1].encode()
root = Path(sys.argv[1])
venv_rel = sys.argv[2]
old_roots = sorted({arg.encode() for arg in sys.argv[3:] if arg and arg.encode() != new}, key=len, reverse=True)

for rel in (venv_rel, ".uv-python"):
    base = root / rel
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size > 10 * 1024 * 1024:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:4096]:
            continue
        next_data = data
        for old in old_roots:
            next_data = next_data.replace(old, new)
        if next_data != data:
            path.write_bytes(next_data)
PY
}

p2a_stage_local_runtime() {
  local source_root="$1"
  source_root="$(cd "${source_root}" && pwd -P)"
  local venv_rel
  venv_rel="$(p2a_runtime_venv_rel "${source_root}")"

  if ! p2a_stage_enabled; then
    export P2A_SHARED_SRC_ROOT="${P2A_SHARED_SRC_ROOT:-${source_root}}"
    export P2A_RUNTIME_SRC_ROOT="${P2A_RUNTIME_SRC_ROOT:-${source_root}}"
    export P2A_VENV_DIR="${venv_rel}"
    export UV_PROJECT_ENVIRONMENT="${source_root}/${venv_rel}"
    return 0
  fi

  if ! command -v rsync >/dev/null 2>&1; then
    echo "[stage] rsync is required for P2A_STAGE_LOCAL_RUNTIME=1." >&2
    return 2
  fi
  if [[ ! -x "${source_root}/${venv_rel}/bin/python" || ! -x "${source_root}/${venv_rel}/bin/ray" ]]; then
    echo "[stage] missing source runtime under ${source_root}/${venv_rel}." >&2
    return 2
  fi

  local local_root local_src_root stage_dir stamp_file source_stamp
  local_root="$(p2a_abs_dir "${P2A_LOCAL_ROOT:-/tmp/p2a-traceanalyzer}")"
  local_src_root="${P2A_LOCAL_SRC_ROOT:-${local_root}/TraceAnalyzer}"
  local_src_root="$(p2a_abs_dir "${local_src_root}")"

  export P2A_SHARED_SRC_ROOT="${P2A_SHARED_SRC_ROOT:-${source_root}}"
  export P2A_RUNTIME_SRC_ROOT="${local_src_root}"
  export P2A_VENV_DIR="${venv_rel}"
  export UV_PROJECT_ENVIRONMENT="${local_src_root}/${venv_rel}"

  if [[ "${source_root}" == "${local_src_root}" ]]; then
    return 0
  fi

  stage_dir="${local_src_root}/.p2a-stage"
  mkdir -p "${stage_dir}"

  {
    if command -v flock >/dev/null 2>&1; then
      flock 9
    fi

    echo "[stage] source: ${source_root}"
    echo "[stage] runtime: ${local_src_root}"

    rsync -a --delete \
      --exclude='.git/' \
      --exclude='.venv' \
      --exclude='.venv/' \
      --exclude='.venv-cu128' \
      --exclude='.venv-cu128/' \
      --exclude="${venv_rel}" \
      --exclude="${venv_rel}/" \
      --exclude='.uv-python' \
      --exclude='.uv-python/' \
      --exclude='.p2a-stage/' \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      --exclude='.pytest_cache/' \
      --exclude='.ruff_cache/' \
      --exclude='outputs/' \
      --exclude='wandb/' \
      --exclude='ray_results/' \
      --exclude='checkpoints/' \
      "${source_root}/" "${local_src_root}/"

    source_stamp="$(p2a_runtime_stamp "${source_root}" "${venv_rel}")"
    stamp_file="${stage_dir}/runtime.stamp"
    if [[ "${P2A_FORCE_STAGE_LOCAL_RUNTIME:-0}" == "1" || -L "${local_src_root}/${venv_rel}" || -L "${local_src_root}/.uv-python" || ! -f "${stamp_file}" || "$(cat "${stamp_file}")" != "${source_stamp}" ]]; then
      echo "[stage] syncing local Python runtime"
      if [[ -L "${local_src_root}/${venv_rel}" ]]; then
        rm -f "${local_src_root:?}/${venv_rel}"
      fi
      if [[ -L "${local_src_root}/.uv-python" ]]; then
        rm -f "${local_src_root}/.uv-python"
      fi
      mkdir -p "$(dirname "${local_src_root}/${venv_rel}")"
      if [[ -d "${source_root}/.uv-python" ]]; then
        mkdir -p "${local_src_root}/.uv-python"
        rsync -a --delete "${source_root}/.uv-python/" "${local_src_root}/.uv-python/"
      fi
      mkdir -p "${local_src_root}/${venv_rel}"
      rsync -a --delete "${source_root}/${venv_rel}/" "${local_src_root}/${venv_rel}/"
      p2a_rewrite_staged_paths "${source_root}" "${local_src_root}" "${venv_rel}"
      printf '%s\n' "${source_stamp}" > "${stamp_file}"
    else
      echo "[stage] local Python runtime is current"
    fi
  } 9>"${local_root}/TraceAnalyzer.stage.lock"
}
