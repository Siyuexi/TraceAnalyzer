#!/usr/bin/env bash
set -euo pipefail

MASTER_IP="$1"
PORT="${PORT:-6379}"
NUM_GPUS="${NUM_GPUS:-8}"

LOCAL_IPS=$(hostname -I 2>/dev/null || true)

pip install -U "click==8.2.1"

if echo " $LOCAL_IPS " | grep -qw "$MASTER_IP"; then
  echo "[Ray] This node is master: $MASTER_IP"

  ray stop -f || true

  ray start \
    --head \
    --port="$PORT" \
    --node-ip-address="$MASTER_IP" \
    --num-gpus="$NUM_GPUS"

else
  WORKER_IP="${WORKER_IP:-$(ip route get "$MASTER_IP" | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')}"

  echo "[Ray] This node is worker: $WORKER_IP"
  echo "[Ray] Connecting to master: $MASTER_IP:$PORT"

  ray stop -f || true

  ray start \
    --address="$MASTER_IP:$PORT" \
    --node-ip-address="$WORKER_IP" \
    --num-gpus="$NUM_GPUS"
fi