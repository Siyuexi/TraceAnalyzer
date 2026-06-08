#!/usr/bin/env bash
# Debug helper — verify the dependency set installs + imports on a CPU box.
#
# Does NOT touch the [gpu] extra (vllm / flash-attn / megatron-core need CUDA) and does
# NOT touch uv.lock (uses the low-level `uv pip` interface, resolving only what is asked).
#
#   bash scripts/check_deps_cpu.sh           # core only: the data-build + bonus-map pipeline
#                                            #   (datasets / r2e-gym / swebench / arl / swe-rex …)
#   bash scripts/check_deps_cpu.sh --train   # + uni-agent/verl framework (CPU torch — heavy/slow)
#
# Exit code 0 = every checked module imported; non-zero = something is missing/broken.
set -euo pipefail
cd "$(dirname "$0")/.."

WANT_TRAIN=0
[[ "${1:-}" == "--train" ]] && WANT_TRAIN=1

# r2e-gym: git package with a stale datasets==2.19 pin → always --no-deps (build_data r2e needs ParsedCommit).
R2EGYM="git+https://github.com/R2E-Gym/R2E-Gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735"

if [[ $WANT_TRAIN -eq 1 ]]; then
  # [train] pulls verl (editable, via [tool.uv.sources]) + verl's own install_requires.
  echo ">> installing core + [train] (uni-agent + verl editable + framework, CPU torch) ..."
  uv pip install -e '.[train]'
  echo ">> installing r2e-gym (--no-deps, git) ..."
  uv pip install --no-deps "$R2EGYM"
  MODS="datasets r2egym pandas pyarrow numpy arl swerex torch ray hydra transformers accelerate peft tensordict verl uni_agent"
else
  # core pulls uni-agent (editable, via [tool.uv.sources]); no torch/framework.
  echo ">> installing core (data-build + bonus-map pipeline + uni-agent) ..."
  uv pip install -e .
  echo ">> installing r2e-gym (--no-deps, git) ..."
  uv pip install --no-deps "$R2EGYM"
  MODS="datasets r2egym pandas pyarrow numpy arl swerex uni_agent"
fi

echo ">> smoke-importing: $MODS"
uv run --no-sync python - "$MODS" <<'PY'
import importlib, sys
mods = sys.argv[1].split()
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001 - report, don't raise
        bad.append(f"{m}: {type(e).__name__}: {str(e)[:120]}")
print("checked:", " ".join(mods))
if bad:
    print("MISSING / BROKEN:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("CPU dependency check: OK")
PY
