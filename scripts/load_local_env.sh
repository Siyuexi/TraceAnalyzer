#!/usr/bin/env bash
# Load private machine-local endpoints and credentials.

p2a_local_env_file() {
  local root="${1:-}"
  if [[ -z "${root}" ]]; then
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
  printf '%s\n' "${P2A_LOCAL_ENV_FILE:-${root}/.secrets/ips.sh}"
}

p2a_source_local_env() {
  local root="${1:-}"
  local env_file
  env_file="$(p2a_local_env_file "${root}")"
  if [[ -f "${env_file}" ]]; then
    # shellcheck disable=SC1090
    source "${env_file}"
  fi
}

p2a_require_env() {
  local key="$1"
  if [[ -z "${!key:-}" ]]; then
    echo "[local-env] ${key} is required; set it or source .secrets/ips.sh." >&2
    return 2
  fi
}
