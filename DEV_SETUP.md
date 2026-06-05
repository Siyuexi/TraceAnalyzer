# Local dev / dependency setup (ARL migration)

Dependency model mirrors uni-agent's own (`uni-agent/README.md` → Installation):
uni-agent + its nested **verl** are editable `--no-deps` installs; a short list of
**light runtime deps** is installed on top; the heavy **GPU/training stack**
(`uni-agent/verl/requirements.txt`: torch / vllm / megatron / flash-attn …) is a
*separate* step done **only on GPU**.

We add exactly one thing on top: the **ARL SDK** (`arl-env==0.3.1`, providing
`SandboxSession` + `InteractiveShellClient`). `src/pyproject.toml` declares only
the GPU-free light deps + `arl-env`.

## Layout (nested submodules — do NOT modify them)
```
root(git) → src/ (submodule) → uni-agent/ (submodule) → verl/ (submodule)
```
`env/` (ARL backend) and `p2a/` (bonus-map) are our code; uni-agent stays unmodified.

## Local (CPU, no GPU) — for ARL runtime + adapter tests + live smoke
```bash
cd src
uv sync                                   # installs arl-env + light deps (NO torch)
uv pip install --no-deps -e ./uni-agent/verl
uv pip install --no-deps -e ./uni-agent
# run with the framework on PYTHONPATH (as the scripts already do):
PYTHONPATH=uni-agent/verl:uni-agent:. uv run python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=uni-agent/verl:uni-agent:. uv run python -m env.smoke --image <r2e-image>
```

## GPU (training) — add the heavy stack
```bash
cd src/uni-agent
pip install -r verl/requirements.txt      # torch / vllm / megatron / flash-attn …
```

## ⚠️ Temporary local GPU-dep skips — DO NOT COMMIT, REVERT BEFORE GPU
If a transitive resolve drags a GPU-only package onto a CPU box, skip it locally and
**log it here**, then undo before any GPU run. None of these edits go into git.

| date | what was skipped/changed (local only) | reverted before GPU? |
|------|----------------------------------------|----------------------|
| _(none yet)_ | | |

Rule: temporary CPU-only hacks live in the working tree only; `git status` must be
clean of them before a GPU run. The committed truth is: full stack installable on GPU.
