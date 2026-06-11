#!/usr/bin/env bash
# Stage the shared checkout and Python runtime onto node-local disk.

p2a_stage_enabled() {
  [[ "${P2A_STAGE_LOCAL_RUNTIME:-${P2A_LOCAL_RUNTIME:-0}}" == "1" ]]
}

p2a_abs_dir() {
  local path="$1"
  mkdir -p "${path}"
  (cd "${path}" && pwd -P)
}

p2a_runtime_stamp() {
  local source_root="$1"
  (
    cd "${source_root}"
    printf 'source=%s\n' "${source_root}"
    git rev-parse HEAD 2>/dev/null || true
    git submodule status --recursive 2>/dev/null || true
    for path in pyproject.toml uv.lock .venv/pyvenv.cfg .venv/bin/python .venv/bin/ray; do
      if [[ -e "${path}" ]]; then
        stat -c '%n %s %Y' "${path}" 2>/dev/null || ls -l "${path}"
      fi
    done
  ) | sha256sum | awk '{print $1}'
}

p2a_rewrite_staged_paths() {
  local old_root="$1"
  local new_root="$2"
  local patch_python
  local roots=("${old_root}")
  patch_python="$(command -v python3 || command -v python || true)"
  if [[ -z "${patch_python}" ]]; then
    echo "[stage] python3 or python is required to rewrite staged venv paths." >&2
    return 2
  fi

  if [[ -f "${new_root}/.venv/bin/ray" ]]; then
    local shebang
    IFS= read -r shebang < "${new_root}/.venv/bin/ray" || true
    shebang="${shebang#\#!}"
    if [[ "${shebang}" == */.venv/bin/python* ]]; then
      roots+=("${shebang%%/.venv/bin/python*}")
    fi
  fi
  if [[ -f "${new_root}/.venv/pyvenv.cfg" ]]; then
    local line root
    while IFS= read -r line; do
      if [[ "${line}" == *"/.uv-python/"* ]]; then
        root="${line#*= }"
        roots+=("${root%%/.uv-python/*}")
      fi
    done < "${new_root}/.venv/pyvenv.cfg"
  fi
  if [[ -L "${new_root}/.venv" || -L "${new_root}/.uv-python" ]]; then
    echo "[stage] staged .venv/.uv-python must be real local directories, not symlinks." >&2
    return 2
  fi

  "${patch_python}" - "${new_root}" "${roots[@]}" <<'PY'
import sys
from pathlib import Path

new = sys.argv[1].encode()
root = Path(sys.argv[1])
old_roots = sorted({arg.encode() for arg in sys.argv[2:] if arg and arg.encode() != new}, key=len, reverse=True)

for rel in (".venv", ".uv-python"):
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

  if ! p2a_stage_enabled; then
    export P2A_SHARED_SRC_ROOT="${P2A_SHARED_SRC_ROOT:-${source_root}}"
    export P2A_RUNTIME_SRC_ROOT="${P2A_RUNTIME_SRC_ROOT:-${source_root}}"
    return 0
  fi

  if ! command -v rsync >/dev/null 2>&1; then
    echo "[stage] rsync is required for P2A_STAGE_LOCAL_RUNTIME=1." >&2
    return 2
  fi
  if [[ ! -x "${source_root}/.venv/bin/python" || ! -x "${source_root}/.venv/bin/ray" ]]; then
    echo "[stage] missing source runtime under ${source_root}/.venv." >&2
    return 2
  fi

  local local_root local_src_root stage_dir stamp_file source_stamp
  local_root="$(p2a_abs_dir "${P2A_LOCAL_ROOT:-/tmp/p2a-traceanalyzer}")"
  local_src_root="${P2A_LOCAL_SRC_ROOT:-${local_root}/TraceAnalyzer}"
  local_src_root="$(p2a_abs_dir "${local_src_root}")"

  export P2A_SHARED_SRC_ROOT="${source_root}"
  export P2A_RUNTIME_SRC_ROOT="${local_src_root}"
  export UV_PROJECT_ENVIRONMENT="${local_src_root}/.venv"

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

    source_stamp="$(p2a_runtime_stamp "${source_root}")"
    stamp_file="${stage_dir}/runtime.stamp"
    if [[ "${P2A_FORCE_STAGE_LOCAL_RUNTIME:-0}" == "1" || -L "${local_src_root}/.venv" || -L "${local_src_root}/.uv-python" || ! -f "${stamp_file}" || "$(cat "${stamp_file}")" != "${source_stamp}" ]]; then
      echo "[stage] syncing local Python runtime"
      if [[ -L "${local_src_root}/.venv" ]]; then
        rm -f "${local_src_root}/.venv"
      fi
      if [[ -L "${local_src_root}/.uv-python" ]]; then
        rm -f "${local_src_root}/.uv-python"
      fi
      mkdir -p "${local_src_root}/.uv-python" "${local_src_root}/.venv"
      rsync -a --delete "${source_root}/.uv-python/" "${local_src_root}/.uv-python/"
      rsync -a --delete "${source_root}/.venv/" "${local_src_root}/.venv/"
      p2a_rewrite_staged_paths "${source_root}" "${local_src_root}"
      printf '%s\n' "${source_stamp}" > "${stamp_file}"
    else
      echo "[stage] local Python runtime is current"
    fi
  } 9>"${local_root}/TraceAnalyzer.stage.lock"
}
