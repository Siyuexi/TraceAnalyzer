#!/usr/bin/env bash
# Build a Uni-Agent-compatible cu128 runtime for the fused Megatron/mbridge path.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_uni_agent_cu128_runtime.sh

Builds a separate pip-managed runtime under .venv-cu128 by default. The normal
uv-managed .venv remains available for the existing cu130 lock.

Key overrides:
  P2A_CU128_VENV_DIR=.venv-cu128
  P2A_CU128_CUDA_HOME=/usr/local/cuda-12.8
  P2A_CU128_PYTHON=/path/to/python3.11
  P2A_CU128_REBUILD=1
  P2A_CU128_VLLM_SPEC=vllm==0.11.0  # pip spec installed --no-deps; "source" builds from git
  P2A_CU128_VLLM_REF=v0.11.0        # git ref when P2A_CU128_VLLM_SPEC=source
  P2A_CU128_INSTALL_APEX=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${SRC_ROOT}"

VENV_REL="${P2A_CU128_VENV_DIR:-${P2A_VENV_DIR:-.venv-cu128}}"
VENV_REL="${VENV_REL#./}"
VENV_REL="${VENV_REL%/}"
case "${VENV_REL}" in
  ""|/*|../*|*/../*|*/..)
    echo "[cu128] Runtime venv must be a relative path under ${SRC_ROOT}: ${VENV_REL}" >&2
    exit 2
    ;;
esac
VENV_PATH="${SRC_ROOT}/${VENV_REL}"
BUILD_ROOT="${P2A_CU128_BUILD_ROOT:-${SRC_ROOT}/.p2a-build/cu128}"

cuda_release() {
  local cuda_home="$1"
  if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    return 1
  fi
  "${cuda_home}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1
}

choose_cuda_home() {
  local candidate release
  if [[ -n "${P2A_CU128_CUDA_HOME:-}" ]]; then
    candidate="${P2A_CU128_CUDA_HOME}"
    release="$(cuda_release "${candidate}" || true)"
    if [[ "${release}" != "12.8" ]]; then
      echo "[cu128] ${candidate} is not a CUDA 12.8 toolkit (nvcc release=${release:-missing})." >&2
      exit 2
    fi
    printf '%s\n' "${candidate}"
    return
  fi
  for candidate in "${CUDA_HOME:-}" /usr/local/cuda-12.8 /usr/local/cuda; do
    [[ -n "${candidate}" ]] || continue
    release="$(cuda_release "${candidate}" || true)"
    if [[ "${release}" == "12.8" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  echo "[cu128] CUDA 12.8 toolkit not found. Install it side-by-side and set P2A_CU128_CUDA_HOME=/path/to/cuda-12.8." >&2
  exit 2
}

choose_python() {
  local python_bin
  if [[ -n "${P2A_CU128_PYTHON:-}" ]]; then
    python_bin="${P2A_CU128_PYTHON}"
  elif command -v python3.11 >/dev/null 2>&1; then
    python_bin="$(command -v python3.11)"
  elif command -v uv >/dev/null 2>&1; then
    uv python install --managed-python 3.11 >&2
    python_bin="$(uv python find --managed-python --no-project 3.11)"
  else
    echo "[cu128] python3.11 or uv is required to build ${VENV_REL}." >&2
    exit 2
  fi
  # venv records home= from the invoking binary's directory; a symlinked
  # python3.11 makes the --copies interpreter unable to locate its stdlib.
  readlink -f "${python_bin}"
}

require_submodules() {
  if [[ ! -d "${SRC_ROOT}/uni-agent/uni_agent" || ! -d "${SRC_ROOT}/uni-agent/verl/verl" ]]; then
    echo "[cu128] Missing Uni-Agent submodules. Run: git submodule update --init --recursive" >&2
    exit 2
  fi
}

write_env_profile() {
  local cuda_home="$1"
  cat > "${VENV_PATH}/p2a-cu128.env" <<EOF
_p2a_profile_path="\${BASH_SOURCE[0]:-\${0}}"
_p2a_venv_path="\$(cd "\$(dirname "\${_p2a_profile_path}")" && pwd -P)"
export P2A_VENV_DIR="\$(basename "\${_p2a_venv_path}")"
export UV_PROJECT_ENVIRONMENT="\${_p2a_venv_path}"
export VIRTUAL_ENV="\${_p2a_venv_path}"
export CUDA_HOME="${cuda_home}"
export CUDA_PATH="${cuda_home}"
export PATH="\${_p2a_venv_path}/bin:${cuda_home}/bin:\${PATH}"
export LD_LIBRARY_PATH="${cuda_home}/lib64:\${LD_LIBRARY_PATH:-}"
export CPATH="${cuda_home}/include:\${CPATH:-}"
export LIBRARY_PATH="${cuda_home}/lib64:\${LIBRARY_PATH:-}"
export NVTE_FRAMEWORK="pytorch"
export MAX_JOBS="${MAX_JOBS:-32}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
unset _p2a_profile_path _p2a_venv_path
EOF
}

install_editables_and_path() {
  local python_bin="$1"
  local site_packages
  "${python_bin}" -m pip install --no-deps -e "${SRC_ROOT}/uni-agent/verl" -e "${SRC_ROOT}/uni-agent" -e "${SRC_ROOT}"
  site_packages="$("${python_bin}" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else site.getusersitepackages())
PY
)"
  cat > "${site_packages}/p2a-traceanalyzer.pth" <<EOF
${SRC_ROOT}
${SRC_ROOT}/uni-agent
${SRC_ROOT}/uni-agent/verl
EOF
}

install_megatron() {
  local python_bin="$1"
  local repo="${BUILD_ROOT}/Megatron-LM"
  local ref="${P2A_CU128_MEGATRON_REF:-core_v0.13.0}"
  mkdir -p "${BUILD_ROOT}"
  if [[ ! -d "${repo}/.git" ]]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git "${repo}"
  fi
  git -C "${repo}" fetch --tags origin "${ref}"
  git -C "${repo}" checkout "${ref}"
  "${python_bin}" -m pip install --no-deps -e "${repo}"
}

install_vllm() {
  local python_bin="$1"
  local repo="${BUILD_ROOT}/vllm"
  local ref="${P2A_CU128_VLLM_REF:-v0.11.0}"
  local spec="${P2A_CU128_VLLM_SPEC:-vllm==0.11.0}"

  # vllm itself is installed --no-deps to keep the torch cu128 pins authoritative,
  # so its runtime dependency closure (vllm 0.11.0 requires_dist) is installed here.
  "${python_bin}" -m pip install --no-cache-dir \
    "cbor2" \
    "setproctitle" \
    "blake3" \
    "openai_harmony" \
    "pybase64" \
    "msgspec" \
    "partial_json_parser" \
    "py-cpuinfo" \
    "diskcache==5.6.3" \
    "cachetools" \
    "sentencepiece" \
    "openai>=1.99.1" \
    "pillow" \
    "prometheus_client>=0.18.0" \
    "prometheus-fastapi-instrumentator>=7.0.0" \
    "tiktoken>=0.6.0" \
    "lm-format-enforcer==0.11.3" \
    "llguidance>=0.7.11,<0.8.0" \
    "outlines_core==0.2.11" \
    "lark==1.2.2" \
    "gguf>=0.13.0" \
    "mistral_common[audio,image]>=1.8.2" \
    "opencv-python-headless>=4.11.0" \
    "einops" \
    "compressed-tensors==0.11.0" \
    "depyf==0.19.0" \
    "cloudpickle" \
    "watchfiles" \
    "python-json-logger" \
    "scipy" \
    "numba==0.61.2" \
    "pyzmq>=25.0.0" \
    "regex" \
    "psutil" \
    "cupy-cuda12x"
  "${python_bin}" -m pip install --no-deps --no-cache-dir "xformers==0.0.32.post1"

  if [[ "${spec}" != "source" ]]; then
    "${python_bin}" -m pip install --no-deps --no-build-isolation --no-cache-dir "${spec}"
    return
  fi

  mkdir -p "${BUILD_ROOT}"
  if [[ ! -d "${repo}/.git" ]]; then
    git clone https://github.com/vllm-project/vllm.git "${repo}"
  fi
  git -C "${repo}" fetch --tags origin "${ref}"
  git -C "${repo}" checkout "${ref}"
  "${python_bin}" -m pip install --no-deps --no-build-isolation --no-cache-dir -e "${repo}"
}

verify_runtime() {
  local python_bin="$1"
  "${python_bin}" - <<'PY'
import importlib
import os
import sys

def version(name):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"IMPORT-FAILED {type(exc).__name__}: {exc}"
    return getattr(module, "__version__", "unknown")

print(f"[cu128] python={sys.executable}")
print(f"[cu128] CUDA_HOME={os.environ.get('CUDA_HOME', '')}")
print(f"[cu128] torch={version('torch')} torch_cuda={getattr(importlib.import_module('torch').version, 'cuda', '')}")
for name in ("vllm", "flash_attn", "transformer_engine", "megatron.core", "mbridge", "ray"):
    print(f"[cu128] {name}={version(name)}")
import transformer_engine.pytorch  # noqa: F401
from verl.workers.engine import EngineRegistry, MegatronEngine
registered = sorted(EngineRegistry._engines.get("language_model", {}))
if MegatronEngine is None or "megatron" not in registered:
    importlib.import_module("verl.workers.engine.megatron")
    registered = sorted(EngineRegistry._engines.get("language_model", {}))
if "megatron" not in registered:
    raise SystemExit(f"[cu128] megatron backend is not registered: {registered}")
print(f"[cu128] megatron backend registered: {registered}")
PY
}

require_submodules
CUDA_HOME_SELECTED="$(choose_cuda_home)"
PYTHON_BASE="$(choose_python)"

if [[ "${P2A_CU128_REBUILD:-0}" == "1" || ! -x "${VENV_PATH}/bin/python" ]]; then
  "${PYTHON_BASE}" -m venv --clear --copies "${VENV_PATH}"
fi

PYTHON_BIN="${VENV_PATH}/bin/python"
write_env_profile "${CUDA_HOME_SELECTED}"
# shellcheck disable=SC1091
source "${VENV_PATH}/p2a-cu128.env"

export PIP_NO_BUILD_ISOLATION="${PIP_NO_BUILD_ISOLATION:-0}"
export PIP_ROOT_USER_ACTION="${PIP_ROOT_USER_ACTION:-ignore}"
export PIP_CONSTRAINT=""

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel packaging ninja cmake setuptools_scm

"${PYTHON_BIN}" -m pip install --no-cache-dir --index-url "${P2A_CU128_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}" \
  "${P2A_CU128_TORCH_SPEC:-torch==2.8.0}" \
  "${P2A_CU128_TORCHVISION_SPEC:-torchvision==0.23.0}" \
  "${P2A_CU128_TORCHAUDIO_SPEC:-torchaudio==2.8.0}"

"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "accelerate" \
  "codetiming" \
  "datasets" \
  "dill" \
  "hydra-core" \
  "numpy<2.0.0" \
  "pandas" \
  "peft" \
  "pyarrow>=19.0.1" \
  "pybind11" \
  "pylatexenc" \
  "ray[default]>=2.41.0" \
  "torchdata" \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
  "transformers[hf_xet]==4.55.4" \
  "wandb" \
  "tensorboard" \
  "liger-kernel" \
  "mathruler" \
  "math-verify" \
  "qwen-vl-utils" \
  "blobfile" \
  "xgrammar" \
  "pytest" \
  "py-spy" \
  "pre-commit" \
  "ruff" \
  "arl-env==0.3.1" \
  "swe-rex>=1.4.0" \
  "websockets" \
  "loguru" \
  "pydantic" \
  "pydantic-settings" \
  "aiohttp" \
  "httpx" \
  "pyyaml" \
  "tqdm" \
  "huggingface-hub" \
  "orjson" \
  "nvidia-ml-py>=12.560.30" \
  "fastapi[standard]>=0.115.0" \
  "optree>=0.13.0" \
  "grpcio>=1.62.1"

"${PYTHON_BIN}" -m pip install --no-cache-dir --no-deps \
  "git+https://github.com/R2E-Gym/R2E-Gym.git@${P2A_R2E_GYM_REF:-0d94c4eb9431cd195c55a7ea3abd54006c9a1735}"

install_vllm "${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --no-cache-dir --no-build-isolation "${P2A_CU128_FLASH_ATTN_SPEC:-flash_attn==2.7.4.post1}"

if [[ "${P2A_CU128_INSTALL_APEX:-1}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    "git+https://github.com/NVIDIA/apex.git"
fi

NVTE_FRAMEWORK=pytorch "${PYTHON_BIN}" -m pip install --no-deps --no-cache-dir --no-build-isolation \
  "git+https://github.com/NVIDIA/TransformerEngine.git@${P2A_CU128_TE_REF:-v2.2.1}"

install_megatron "${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --no-cache-dir "${P2A_CU128_MBRIDGE_SPEC:-git+https://github.com/ISEEKYAN/mbridge.git}"
install_editables_and_path "${PYTHON_BIN}"
verify_runtime "${PYTHON_BIN}"

cat <<EOF
[cu128] Runtime ready:
  venv: ${VENV_PATH}
  CUDA_HOME: ${CUDA_HOME_SELECTED}
  activate: source ${VENV_PATH}/p2a-cu128.env
  launch: P2A_VENV_DIR=${VENV_REL} bash scripts/main.sh
EOF
