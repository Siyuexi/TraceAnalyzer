#!/usr/bin/env bash
# Run the issue #27 OpenAI-compatible inference-only SWE harness.
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SRC_ROOT}"

export PYTHONPATH="${PYTHONPATH:-uni-agent/verl:uni-agent:.}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export P2A_DEPLOYMENT="${P2A_DEPLOYMENT:-arl}"
export UNI_AGENT_P2A_TRACE="${UNI_AGENT_P2A_TRACE:-1}"

exec uv run python -m p2a.third_party_eval "$@"
