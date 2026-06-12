# Dependency setup (ARL migration)

All dependencies are declared in `pyproject.toml`, and the **training framework deps are
sourced from verl itself** — not hand-copied. `[tool.uv.sources]` installs the `uni-agent`
+ `verl` submodules editable, so listing `verl` pulls verl's own `install_requires` /
`extras_require` straight from `uni-agent/verl/setup.py`. No drift, no duplicated pins.

| install | what | runs on |
|---------|------|---------|
| `uv sync --locked` | core: ARL runtime + data build + precompute + `uni-agent` + `r2e-gym` | **CPU** |
| `uv sync --locked --extra train` | + `verl` (its framework deps: accelerate/ray/tensordict/numpy<2/...) | CPU (smoke) / GPU |
| `uv sync --locked --extra train --extra gpu` | + verl's GPU extras (`verl[vllm,gpu,geo,mcore,math]`) | **GPU only** |

`uni-agent` declares zero deps; `verl` is the source of truth for the framework (base +
GPU). `r2e-gym` is a pinned Git source in `pyproject.toml`; uv metadata override keeps
it no-deps so its upstream `datasets==2.19` pin does not override this repo's data stack.

`pyproject.toml` is **deps-only** (`[tool.setuptools] packages = []`): `p2a/`, `env/`,
`scripts/` stay importable via `PYTHONPATH=uni-agent/verl:uni-agent:.` (as the launchers set).

## Layout (nested submodules — do NOT modify them)
```
root(git) → src/ (submodule) → uni-agent/ (submodule) → verl/ (submodule)
```

## Local (CPU) — verify deps install + import
```bash
cd src
bash scripts/check_deps_cpu.sh          # core + r2e-gym, smoke-imports, prints OK/FAIL
bash scripts/check_deps_cpu.sh --train  # + verl framework (heavy; downgrades numpy to <2)
```

## GPU (training)
```bash
cd src
uv sync --locked --extra train --extra gpu
```
GPU pins (vllm 0.8.5–0.12.0, flash-attn, torchvision, mbridge for Megatron, …) all come
from verl's own `extras_require`. vllm/flash-attn need a CUDA toolchain on the node.
Note: on a CPU-only `[train]` resolve, `transformers` floats to the latest; on the GPU
`[train,gpu]` install, verl's `vllm` extra constrains it to a vllm-compatible version.

### Fused Megatron cu128 runtime

For the Uni-Agent fused Megatron/mbridge path, use a separate pip-managed
runtime instead of the cu130 `uv.lock` environment:

```bash
cd src
export P2A_CU128_CUDA_HOME=/usr/local/cuda-12.8
bash scripts/setup_uni_agent_cu128_runtime.sh
source .venv-cu128/p2a-cu128.env
python scripts/check_uni_agent_runtime.py
```

This creates `.venv-cu128` with torch 2.8.0/cu128, flash-attn 2.7.4.post1,
TransformerEngine v2.2.1, Megatron-LM core_v0.13.0, and mbridge. The training
launchers prefer `.venv-cu128` when it exists and stage that exact venv to Ray
workers. Do not run `uv sync` into `.venv-cu128`; rebuild it with the setup
script if the cu128 stack changes.

## Shared HuggingFace assets

By default, scripts share HuggingFace assets two levels above `src/`:

- Models: `../../models/<repo-name>`
- Datasets: `../../datasets/<repo-name>/<split>`

The default training model is `Qwen/Qwen3-Coder-30B-A3B-Instruct`, saved as
`../../models/Qwen3-Coder-30B-A3B-Instruct`. Existing local directories are reused;
missing assets are downloaded and saved there. Override with `MODEL_PATH`,
`P2A_MODEL_REPO`, `P2A_MODELS_DIR`, `P2A_DATASETS_DIR`, or `P2A_SHARED_ROOT`.
