#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/ray_setup.sh <HEAD_IP> [start|stop|smoke|restart-cluster]

Modes:
  start  Stop any local Ray runtime and start this node. This is the default.
  stop   Stop the local Ray runtime only.
  smoke  Do not restart Ray; submit a tiny Ray Jobs task to verify the dashboard.
  restart-cluster
         From the head node, stop workers first, then head, then start head,
         start workers, and run the smoke check. Requires RAY_WORKER_HOSTS.

Common overrides:
  RAY_GCS_PORT=6379
  RAY_DASHBOARD_PORT=8265
  UV_PROJECT_ENVIRONMENT=$PWD/.venv
  P2A_LOCAL_RUNTIME=0|1
  P2A_LOCAL_ROOT=/tmp/p2a-traceanalyzer
  P2A_RAY_LAUNCHER=venv|native
  P2A_SYNC_NATIVE_RAY=0|1
  P2A_SYNC_LOCAL_VENV=0|1
  NUM_GPUS=8
  NUM_CPUS=64
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

SHARED_SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${SHARED_SRC_ROOT}"
cd "${SHARED_SRC_ROOT}"

MASTER_IP="$1"
if [[ -z "${MASTER_IP}" ]]; then
  echo "[Ray] HEAD_IP must not be empty." >&2
  usage
  exit 2
fi

MODE="${2:-start}"
case "${MODE}" in
  start|stop|smoke|restart-cluster) ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ -n "${PORT:-}" && -z "${RAY_GCS_PORT:-}" && -z "${RAY_PORT:-}" ]]; then
  echo "[Ray] Ignoring generic PORT=${PORT}; set RAY_GCS_PORT to override Ray's GCS port." >&2
fi
RAY_GCS_PORT="${RAY_GCS_PORT:-${RAY_PORT:-6379}}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_CPUS="${NUM_CPUS:-64}"
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
P2A_LOCAL_RUNTIME="${P2A_LOCAL_RUNTIME:-0}"
P2A_LOCAL_ROOT="${P2A_LOCAL_ROOT:-/tmp/p2a-traceanalyzer}"
P2A_LOCAL_SRC_ROOT="${P2A_LOCAL_SRC_ROOT:-${P2A_LOCAL_ROOT}/TraceAnalyzer}"
P2A_SYNC_NATIVE_RAY="${P2A_SYNC_NATIVE_RAY:-0}"
P2A_SYNC_LOCAL_VENV="${P2A_SYNC_LOCAL_VENV:-0}"
P2A_REBUILD_LOCAL_VENV="${P2A_REBUILD_LOCAL_VENV:-0}"
P2A_RAY_LAUNCHER="${P2A_RAY_LAUNCHER:-venv}"
P2A_ORIG_HTTP_PROXY="${HTTP_PROXY:-}"
P2A_ORIG_HTTPS_PROXY="${HTTPS_PROXY:-}"
P2A_ORIG_ALL_PROXY="${ALL_PROXY:-}"
P2A_ORIG_http_proxy="${http_proxy:-}"
P2A_ORIG_https_proxy="${https_proxy:-}"
P2A_ORIG_all_proxy="${all_proxy:-}"

SHARED_UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-"${SHARED_SRC_ROOT}/.venv"}
if [[ "${SHARED_UV_PROJECT_ENVIRONMENT}" != /* ]]; then
  SHARED_UV_PROJECT_ENVIRONMENT="${SHARED_SRC_ROOT}/${SHARED_UV_PROJECT_ENVIRONMENT}"
fi
SHARED_PYTHON_BIN="${SHARED_UV_PROJECT_ENVIRONMENT}/bin/python"
SHARED_RAY_BIN="${SHARED_UV_PROJECT_ENVIRONMENT}/bin/ray"

if [[ "${RAY_KEEP_PROXY:-0}" != "1" ]]; then
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
fi
NO_PROXY_APPEND="localhost,127.0.0.1,::1,${MASTER_IP}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${NO_PROXY_APPEND}"
export no_proxy="${no_proxy:+${no_proxy},}${NO_PROXY_APPEND}"

echo "[Ray] shared source: ${SHARED_SRC_ROOT}"
echo "[Ray] shared UV_PROJECT_ENVIRONMENT=${SHARED_UV_PROJECT_ENVIRONMENT}"

if [[ ! -x "${SHARED_PYTHON_BIN}" || ! -x "${SHARED_RAY_BIN}" ]]; then
  echo "[Ray] Missing ${SHARED_PYTHON_BIN} or ${SHARED_RAY_BIN}" >&2
  echo "[Ray] Build the shared src/.venv first, then rerun this script." >&2
  exit 2
fi

LOCAL_IPS=$(hostname -I 2>/dev/null || true)
IS_MASTER=0
if echo " ${LOCAL_IPS} " | grep -qw "${MASTER_IP}"; then
  IS_MASTER=1
fi

if ! SHARED_RAY_VERSION="$("${SHARED_PYTHON_BIN}" - <<'PY'
import sys
import ray

print(f"[Ray] launcher python: {sys.version.split()[0]} ({sys.executable})")
print(f"[Ray] launcher ray: {ray.__version__}")
print(ray.__version__)
PY
)"; then
  echo "[Ray] Ray is not importable from ${SHARED_PYTHON_BIN}" >&2
  echo "[Ray] Run: UV_PROJECT_ENVIRONMENT=\$PWD/.venv uv sync --locked --extra train --extra gpu" >&2
  exit 2
fi
printf '%s\n' "${SHARED_RAY_VERSION}" | sed -n '1,2p'
SHARED_RAY_VERSION="$(printf '%s\n' "${SHARED_RAY_VERSION}" | tail -n 1)"
SHARED_PYTHON_ABI="$("${SHARED_PYTHON_BIN}" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

sync_local_source() {
  if [[ "${P2A_LOCAL_RUNTIME}" != "1" ]]; then
    return 0
  fi
  echo "[Ray] Syncing source to local runtime: ${P2A_LOCAL_SRC_ROOT}"
  mkdir -p "${P2A_LOCAL_SRC_ROOT}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete \
      --exclude='.git/' \
      --exclude='.venv/' \
      --exclude='.uv-python/' \
      --exclude='outputs/' \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      "${SHARED_SRC_ROOT}/" "${P2A_LOCAL_SRC_ROOT}/"
  else
    tar -C "${SHARED_SRC_ROOT}" \
      --exclude='./.git' \
      --exclude='./.venv' \
      --exclude='./.uv-python' \
      --exclude='./outputs' \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      -cf - . | tar -C "${P2A_LOCAL_SRC_ROOT}" -xf -
  fi
  SRC_ROOT="${P2A_LOCAL_SRC_ROOT}"
}

ensure_local_venv() {
  if [[ "${P2A_LOCAL_RUNTIME}" != "1" || "${P2A_SYNC_LOCAL_VENV}" != "1" ]]; then
    return 0
  fi
  local local_venv="${P2A_LOCAL_SRC_ROOT}/.venv"
  if [[ "${P2A_REBUILD_LOCAL_VENV}" != "1" ]]; then
    echo "[Ray] Syncing shared Python and venv to local disk: ${local_venv}"
    mkdir -p "${P2A_LOCAL_SRC_ROOT}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "${SHARED_SRC_ROOT}/.uv-python/" "${P2A_LOCAL_SRC_ROOT}/.uv-python/"
      rsync -a --delete "${SHARED_UV_PROJECT_ENVIRONMENT}/" "${local_venv}/"
    else
      rm -rf "${P2A_LOCAL_SRC_ROOT}/.uv-python" "${local_venv}"
      mkdir -p "${P2A_LOCAL_SRC_ROOT}/.uv-python" "${local_venv}"
      tar -C "${SHARED_SRC_ROOT}/.uv-python" -cf - . | tar -C "${P2A_LOCAL_SRC_ROOT}/.uv-python" -xf -
      tar -C "${SHARED_UV_PROJECT_ENVIRONMENT}" -cf - . | tar -C "${local_venv}" -xf -
    fi
    local patch_python
    patch_python="$(command -v python3 || command -v python || true)"
    if [[ -z "${patch_python}" ]]; then
      echo "[Ray] python3 or python is required to rewrite copied venv paths." >&2
      exit 2
    fi
    "${patch_python}" - "${SHARED_SRC_ROOT}" "${P2A_LOCAL_SRC_ROOT}" "${local_venv}" <<'PY'
import os
import sys
from pathlib import Path

old, new, venv = (arg.encode() for arg in sys.argv[1:4])
venv_path = Path(sys.argv[3])
for path in venv_path.rglob("*"):
    if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
        continue
    data = path.read_bytes()
    if old not in data or b"\0" in data[:4096]:
        continue
    path.write_bytes(data.replace(old, new))
PY
    "${local_venv}/bin/python" - <<'PY'
import ray
import sys

print(f"[Ray] local venv python: {sys.executable}")
print(f"[Ray] local venv ray: {ray.__version__}")
PY
    return 0
  fi
  local uv_bin
  uv_bin="${UV_BIN:-$(command -v uv || true)}"
  if [[ -z "${uv_bin}" ]]; then
    echo "[Ray] uv is required to build local venv; set UV_BIN or install uv." >&2
    exit 2
  fi
  echo "[Ray] Building local venv: ${local_venv}"
  mkdir -p "${P2A_LOCAL_ROOT}/uv_cache"
  (
    cd "${P2A_LOCAL_SRC_ROOT}"
    export UV_PROJECT_ENVIRONMENT="${local_venv}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-${P2A_LOCAL_ROOT}/uv_cache}"
    export HTTP_PROXY="${P2A_ORIG_HTTP_PROXY}"
    export HTTPS_PROXY="${P2A_ORIG_HTTPS_PROXY}"
    export ALL_PROXY="${P2A_ORIG_ALL_PROXY}"
    export http_proxy="${P2A_ORIG_http_proxy}"
    export https_proxy="${P2A_ORIG_https_proxy}"
    export all_proxy="${P2A_ORIG_all_proxy}"
    "${uv_bin}" sync --locked --extra train --extra gpu
  )
}

ensure_native_ray() {
  local native_python native_ray native_version native_python_abi
  native_python="${NATIVE_PYTHON_BIN:-$(command -v python || true)}"
  native_ray="${NATIVE_RAY_BIN:-$(command -v ray || true)}"
  if [[ -z "${native_python}" ]]; then
    echo "[Ray] Native python not found on PATH; set NATIVE_PYTHON_BIN." >&2
    exit 2
  fi
  native_python_abi="$("${native_python}" - <<'PY'
import sys

print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [[ "${P2A_SYNC_LOCAL_VENV}" == "1" && "${native_python_abi}" != "${SHARED_PYTHON_ABI}" && "${P2A_ALLOW_NATIVE_PYTHON_MISMATCH:-0}" != "1" ]]; then
    echo "[Ray] Native python is ${native_python_abi}, but the training venv is ${SHARED_PYTHON_ABI}." >&2
    echo "[Ray] Ray Jobs cannot safely mix these Python ABIs with runtime_env.py_executable." >&2
    echo "[Ray] Use P2A_RAY_LAUNCHER=venv so Ray starts from the copied local venv, or provide a native Python ${SHARED_PYTHON_ABI}." >&2
    exit 2
  fi
  native_version="$("${native_python}" - <<'PY' 2>/dev/null || true
import ray
print(ray.__version__)
PY
)"
  if [[ "${P2A_SYNC_NATIVE_RAY}" == "1" && "${native_version}" != "${SHARED_RAY_VERSION}" ]]; then
    echo "[Ray] Installing native ray ${SHARED_RAY_VERSION} (was ${native_version:-missing})"
    HTTP_PROXY="${P2A_ORIG_HTTP_PROXY}" \
      HTTPS_PROXY="${P2A_ORIG_HTTPS_PROXY}" \
      ALL_PROXY="${P2A_ORIG_ALL_PROXY}" \
      http_proxy="${P2A_ORIG_http_proxy}" \
      https_proxy="${P2A_ORIG_https_proxy}" \
      all_proxy="${P2A_ORIG_all_proxy}" \
      "${native_python}" -m pip install -U "ray[default]==${SHARED_RAY_VERSION}"
    native_ray="${NATIVE_RAY_BIN:-$(command -v ray || true)}"
    native_version="$("${native_python}" - <<'PY'
import ray
print(ray.__version__)
PY
)"
  fi
  if [[ -z "${native_ray}" || "${native_version}" != "${SHARED_RAY_VERSION}" ]]; then
    echo "[Ray] Native ray version is '${native_version:-missing}', expected '${SHARED_RAY_VERSION}'." >&2
    echo "[Ray] Set P2A_SYNC_NATIVE_RAY=1 or use P2A_RAY_LAUNCHER=venv." >&2
    exit 2
  fi
  PYTHON_BIN="${native_python}"
  RAY_BIN="${native_ray}"
}

select_launcher() {
  if [[ "${P2A_RAY_LAUNCHER}" == "native" ]]; then
    ensure_native_ray
    export UV_PROJECT_ENVIRONMENT="${SHARED_UV_PROJECT_ENVIRONMENT}"
    export VIRTUAL_ENV="${SHARED_UV_PROJECT_ENVIRONMENT}"
    return 0
  fi
  UV_PROJECT_ENVIRONMENT="${SHARED_UV_PROJECT_ENVIRONMENT}"
  if [[ "${P2A_LOCAL_RUNTIME}" == "1" && "${P2A_SYNC_LOCAL_VENV}" == "1" ]]; then
    UV_PROJECT_ENVIRONMENT="${P2A_LOCAL_SRC_ROOT}/.venv"
  fi
  export UV_PROJECT_ENVIRONMENT
  PYTHON_BIN=${PYTHON_BIN:-"${UV_PROJECT_ENVIRONMENT}/bin/python"}
  RAY_BIN=${RAY_BIN:-"${UV_PROJECT_ENVIRONMENT}/bin/ray"}
  export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
  export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
}

if [[ "${MODE}" != "stop" ]]; then
  sync_local_source
  ensure_local_venv
fi
select_launcher
cd "${SRC_ROOT}"
echo "[Ray] runtime source: ${SRC_ROOT}"
echo "[Ray] ray launcher: ${RAY_BIN}"
echo "[Ray] python launcher: ${PYTHON_BIN}"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
  echo "[Ray] Missing runtime launcher ${PYTHON_BIN} or ${RAY_BIN}" >&2
  exit 2
fi

dashboard_url() {
  if [[ "${IS_MASTER}" == "1" ]]; then
    printf 'http://127.0.0.1:%s\n' "${RAY_DASHBOARD_PORT}"
  else
    printf 'http://%s:%s\n' "${MASTER_IP}" "${RAY_DASHBOARD_PORT}"
  fi
}

wait_for_dashboard() {
  local url="$1"
  local deadline="${RAY_DASHBOARD_WAIT_SECONDS:-45}"
  local start
  start=$(date +%s)
  while true; do
    if "${PYTHON_BIN}" - "${url}" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/api/version", timeout=2) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
    then
      echo "[Ray] Dashboard reachable: ${url}"
      return 0
    fi
    if (( $(date +%s) - start >= deadline )); then
      echo "[Ray] Dashboard not reachable after ${deadline}s: ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

smoke_submit() {
  local url
  url="${RAY_API_SERVER_ADDRESS:-$(dashboard_url)}"
  export RAY_API_SERVER_ADDRESS="${url}"
  wait_for_dashboard "${url}"
  echo "[Ray] Submitting smoke job to ${url}"
  timeout "${RAY_SMOKE_TIMEOUT_SECONDS:-90}" \
    "${RAY_BIN}" job submit --address="${url}" \
      -- "${PYTHON_BIN}" -c 'import socket; print("ray_setup_smoke_ok", socket.gethostname())'
}

stop_local() {
  echo "[Ray] Stopping local Ray runtime"
  "${RAY_BIN}" stop -f || true
}

start_local() {
  if [[ "${IS_MASTER}" == "1" ]]; then
    echo "[Ray] This node is master: $MASTER_IP"

    stop_local

    "${RAY_BIN}" start \
      --head \
      --port="$RAY_GCS_PORT" \
      --dashboard-host="$RAY_DASHBOARD_HOST" \
      --dashboard-port="$RAY_DASHBOARD_PORT" \
      --node-ip-address="$MASTER_IP" \
      --num-cpus="$NUM_CPUS" \
      --num-gpus="$NUM_GPUS"

    wait_for_dashboard "$(dashboard_url)"

  else
    WORKER_IP="${WORKER_IP:-$(ip route get "$MASTER_IP" | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')}"
    if [[ -z "${WORKER_IP}" ]]; then
      echo "[Ray] Could not infer this worker's IP for master ${MASTER_IP}; set WORKER_IP explicitly." >&2
      exit 2
    fi
    export NO_PROXY="${NO_PROXY},${WORKER_IP}"
    export no_proxy="${no_proxy},${WORKER_IP}"

    echo "[Ray] This node is worker: $WORKER_IP"
    echo "[Ray] Connecting to master: $MASTER_IP:$RAY_GCS_PORT"

    stop_local

    "${RAY_BIN}" start \
      --address="$MASTER_IP:$RAY_GCS_PORT" \
      --node-ip-address="$WORKER_IP" \
      --num-cpus="$NUM_CPUS" \
      --num-gpus="$NUM_GPUS"
  fi
}

remote_node() {
  local host="$1"
  local mode="$2"
  # Workers start from the shared source, then optionally sync their own local runtime.
  ssh ${RAY_SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8} \
    "${host}" \
    "cd '${SHARED_SRC_ROOT}' && RAY_GCS_PORT='${RAY_GCS_PORT}' RAY_DASHBOARD_PORT='${RAY_DASHBOARD_PORT}' UV_PROJECT_ENVIRONMENT='${SHARED_UV_PROJECT_ENVIRONMENT}' NUM_GPUS='${NUM_GPUS}' NUM_CPUS='${NUM_CPUS}' P2A_LOCAL_RUNTIME='${P2A_LOCAL_RUNTIME}' P2A_LOCAL_ROOT='${P2A_LOCAL_ROOT}' P2A_LOCAL_SRC_ROOT='${P2A_LOCAL_SRC_ROOT}' P2A_SYNC_NATIVE_RAY='${P2A_SYNC_NATIVE_RAY}' P2A_SYNC_LOCAL_VENV='${P2A_SYNC_LOCAL_VENV}' P2A_REBUILD_LOCAL_VENV='${P2A_REBUILD_LOCAL_VENV}' P2A_RAY_LAUNCHER='${P2A_RAY_LAUNCHER}' bash scripts/ray_setup.sh '${MASTER_IP}' '${mode}'"
}

restart_cluster() {
  if [[ "${IS_MASTER}" != "1" ]]; then
    echo "[Ray] restart-cluster must be run on the head node ${MASTER_IP}." >&2
    exit 2
  fi
  if [[ -z "${RAY_WORKER_HOSTS:-}" ]]; then
    echo "[Ray] restart-cluster requires RAY_WORKER_HOSTS, e.g. 'host1 host2 host3'." >&2
    exit 2
  fi

  echo "[Ray] Restarting cluster with workers: ${RAY_WORKER_HOSTS}"
  for host in ${RAY_WORKER_HOSTS}; do
    echo "[Ray] Stopping worker ${host}"
    remote_node "${host}" stop
  done

  stop_local
  start_local

  for host in ${RAY_WORKER_HOSTS}; do
    echo "[Ray] Starting worker ${host}"
    remote_node "${host}" start
  done

  smoke_submit
}

case "${MODE}" in
  start)
    start_local
    ;;
  stop)
    stop_local
    ;;
  smoke)
    smoke_submit
    ;;
  restart-cluster)
    restart_cluster
    ;;
esac
