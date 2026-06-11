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
  P2A_STAGE_LOCAL_RUNTIME=1
  P2A_LOCAL_ROOT=/tmp/p2a-traceanalyzer
  NUM_GPUS=8
  NUM_CPUS=$(nproc) when staging locally; 64 when P2A_STAGE_LOCAL_RUNTIME=0
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
P2A_STAGE_LOCAL_RUNTIME="${P2A_STAGE_LOCAL_RUNTIME:-1}"
source "${SHARED_SRC_ROOT}/scripts/stage_local_runtime.sh"
NUM_GPUS="${NUM_GPUS:-8}"
if [[ -z "${NUM_CPUS:-}" ]]; then
  if p2a_stage_enabled; then
    NUM_CPUS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '64')"
  else
    NUM_CPUS=64
  fi
fi
RAY_DASHBOARD_HOST="${RAY_DASHBOARD_HOST:-0.0.0.0}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
p2a_stage_local_runtime "${SHARED_SRC_ROOT}"
SRC_ROOT="${P2A_RUNTIME_SRC_ROOT}"
cd "${SRC_ROOT}"
UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-"${SRC_ROOT}/.venv"}
if [[ "${UV_PROJECT_ENVIRONMENT}" != /* ]]; then
  UV_PROJECT_ENVIRONMENT="${SRC_ROOT}/${UV_PROJECT_ENVIRONMENT}"
fi
export UV_PROJECT_ENVIRONMENT
PYTHON_BIN=${PYTHON_BIN:-"${UV_PROJECT_ENVIRONMENT}/bin/python"}
RAY_BIN=${RAY_BIN:-"${UV_PROJECT_ENVIRONMENT}/bin/ray"}
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

NO_PROXY_APPEND="localhost,127.0.0.1,::1,${MASTER_IP}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${NO_PROXY_APPEND}"
export no_proxy="${no_proxy:+${no_proxy},}${NO_PROXY_APPEND}"

echo "[Ray] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "[Ray] NUM_CPUS=${NUM_CPUS}"
if [[ "${SRC_ROOT}" != "${SHARED_SRC_ROOT}" ]]; then
  echo "[Ray] shared source: ${SHARED_SRC_ROOT}"
  echo "[Ray] runtime source: ${SRC_ROOT}"
fi

if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
  echo "[Ray] Missing ${PYTHON_BIN} or ${RAY_BIN}" >&2
  echo "[Ray] Build the shared src/.venv first, then rerun this script." >&2
  exit 2
fi

LOCAL_IPS=$(hostname -I 2>/dev/null || true)
IS_MASTER=0
if echo " ${LOCAL_IPS} " | grep -qw "${MASTER_IP}"; then
  IS_MASTER=1
fi

if ! "${PYTHON_BIN}" - <<'PY'
import sys
import ray

print(f"[Ray] launcher python: {sys.version.split()[0]} ({sys.executable})")
print(f"[Ray] launcher ray: {ray.__version__}")
PY
then
  echo "[Ray] Ray is not importable from ${PYTHON_BIN}" >&2
  echo "[Ray] Run: UV_PROJECT_ENVIRONMENT=\$PWD/.venv uv sync --locked --extra train --extra gpu" >&2
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
  # Workers enter through the shared source, then stage their own local runtime if enabled.
  ssh ${RAY_SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8} \
    "${host}" \
    "cd '${SHARED_SRC_ROOT}' && RAY_GCS_PORT='${RAY_GCS_PORT}' RAY_DASHBOARD_PORT='${RAY_DASHBOARD_PORT}' NUM_GPUS='${NUM_GPUS}' NUM_CPUS='${NUM_CPUS}' P2A_STAGE_LOCAL_RUNTIME='${P2A_STAGE_LOCAL_RUNTIME:-${P2A_LOCAL_RUNTIME:-0}}' P2A_LOCAL_ROOT='${P2A_LOCAL_ROOT:-/tmp/p2a-traceanalyzer}' P2A_LOCAL_SRC_ROOT='${P2A_LOCAL_SRC_ROOT:-}' P2A_FORCE_STAGE_LOCAL_RUNTIME='${P2A_FORCE_STAGE_LOCAL_RUNTIME:-0}' bash scripts/ray_setup.sh '${MASTER_IP}' '${mode}'"
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
