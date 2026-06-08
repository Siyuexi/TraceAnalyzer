# Dependency setup (ARL migration)

All dependencies are declared in `pyproject.toml`, and the **training framework deps are
sourced from verl itself** — not hand-copied. `[tool.uv.sources]` installs the `uni-agent`
+ `verl` submodules editable, so listing `verl` pulls verl's own `install_requires` /
`extras_require` straight from `uni-agent/verl/setup.py`. No drift, no duplicated pins.

| install | what | runs on |
|---------|------|---------|
| `uv pip install -e .`          | core: ARL runtime + data build + precompute + `uni-agent` | **CPU** |
| `uv pip install -e '.[train]'` | + `verl` (its framework deps: accelerate/ray/tensordict/numpy<2/…) | CPU (smoke) / GPU |
| `uv pip install -e '.[train,gpu]'` | + verl's GPU extras (`verl[vllm,gpu,geo,mcore,math]`) | **GPU only** |

`uni-agent` declares zero deps; `verl` is the source of truth for the framework (base +
GPU). The only thing neither covers is **`r2e-gym`** (ParsedCommit for `build_data r2e`):
a git package pinning `datasets==2.19`, installed `--no-deps`:

```bash
uv pip install --no-deps git+https://github.com/R2E-Gym/R2E-Gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735
```

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
uv pip install -e '.[train,gpu]'
uv pip install --no-deps git+https://github.com/R2E-Gym/R2E-Gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735
```
GPU pins (vllm 0.8.5–0.12.0, flash-attn, torchvision, mbridge for Megatron, …) all come
from verl's own `extras_require`. vllm/flash-attn need a CUDA toolchain on the node.
Note: on a CPU-only `[train]` resolve, `transformers` floats to the latest; on the GPU
`[train,gpu]` install, verl's `vllm` extra constrains it to a vllm-compatible version.
```
