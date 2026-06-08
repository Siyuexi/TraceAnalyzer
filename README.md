# P2A on Uni-Agent + ARL

Program-Analysis-based Process Advantage (P2A) for SWE agentic RL, implemented on
the **Uni-Agent** training stack with our **ARL** cluster as the sandbox backend.
P2A reshapes the per-step RL advantage using a precomputed **bonus map** (the
golden runtime call graph from failing test → patched callable): steps whose agent
actions land on the fault-propagation path get a larger advantage.

Everything is **self-contained**: data comes from HuggingFace, images from the
pair-diag mirror of the original R2E images. There is **no dependency on the old
`src-backup` fork**.

Default HuggingFace assets are shared across sibling projects:

| Asset | Default location from `src/` | Override |
|---|---|---|
| Models | `../../models/<repo-name>` | `MODEL_PATH` / `P2A_MODELS_DIR` / `P2A_MODEL_REPO` |
| Datasets | `../../datasets/<repo-name>/<split>` | `P2A_DATASETS_DIR` |

If a dataset/model is already present there, scripts read it directly. If it is
missing, the script downloads it from HuggingFace and saves it under that shared
location.

## Directory map

```
src/
  config/                     # ALL json/yaml config lives here
    startup_fixups.json       #   per-repo test-startup fixups (source patches + dep pins)
    bad_instances.json        #   R2E instances to exclude from training (skip-list)
  scripts/
    build_data.py             # SINGLE data builder, subcommands: r2e | swebench-hard | skip-list
    uni_agent_arl.sh          # prepare/data/smoke/debug launcher (ARL config)
    ray_setup.sh              # bring up Ray
    train_p2a.sh              # training launcher (baseline OR P2A)
  p2a/
    core.py                   # bonus-map load + read->callgraph match + m(d)=m_max^(1-d) multiplier
    trainer.py                # apply_p2a_reshape: capture agent reads -> reshape advantage   [see TODO]
    main.py                   # training entry; P2AFullyAsyncTrainer (vanilla if P2A_BONUS_MAP_DIR unset)
    test_setup.py             # startup_fixup_command(repo) — loads config/startup_fixups.json
    trace.py                  # instrumentation + call-graph build (bonus-map precompute)
    precompute/precompute_bonus_maps.py  # build bonus maps on the ARL backend
    skip_cases.py             # load_skip_ids() from config/bad_instances.json
  env/                        # ARL deployment + runtime (implements swe-rex AbstractRuntime; no swe-rex server)
  uni-agent/                  # UNMODIFIED Uni-Agent submodule (swe-rex is its runtime interface)
```

## Pipeline

```bash
# 1. Build R2E training data from HuggingFace (full + skip-filtered .train.parquet)
#    NOTE: this does NOT use vefaas. Compute backend is ARL (env/agent_config_arl.yaml)
#    and r2e images are pair-diag. PYTHONPATH includes uni-agent/examples/data_preprocess
#    only so build_data.py can reuse its prompt/schema CONSTANTS by import; the script
#    sets DEPLOYMENT internally just to satisfy that import. No vefaas API is called.
PYTHONPATH=.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py r2e --out $DATA/r2e_gym_subset_p2a.parquet
#   -> r2e_gym_subset_p2a.parquet         (full, for bonus-map precompute)
#   -> r2e_gym_subset_p2a.train.parquet   (bad cases excluded, for training/eval)
#   dependency note: r2e-gym is installed by uv.lock; no manual uv pip install is needed.

# 2. Precompute bonus maps on ARL (pair-diag images + faithful startup fixups)
PYTHONPATH=.:uni-agent:uni-agent/verl P2A_DEPLOYMENT=arl ARL_GATEWAY_URL=$ARL \
  uv run python p2a/precompute/precompute_bonus_maps.py \
    $DATA/r2e_gym_subset_p2a.parquet --output_dir $BONUS --mode dynamic --n_parallel 64

# 3. Build the HARD validation subset (cheap eval; full SWE-bench-Verified is too slow)
PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py swebench-hard --out $DATA/swe_bench_verified_hard.parquet

# 4. Train.  Baseline (no P2A): leave P2A_BONUS_MAP_DIR unset.  P2A: set it.
TRAIN_FILE=$DATA/r2e_gym_subset_p2a.train.parquet TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  MODEL_PATH=$MODEL P2A_BONUS_MAP_DIR=$BONUS P2A_M_MAX=3.0 bash scripts/train_p2a.sh
```

## What you configure yourself

These are knobs you set; the repo does not pin them:

| What | Where |
|---|---|
| Model | `MODEL_PATH` env var; default is `../../models/Qwen3-Coder-30B-A3B-Instruct` from `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Train / val data | `TRAIN_FILE` / `TEST_FILE` env vars (point at the parquets built above) |
| GPU layout | `NNODES_TRAIN` / `NNODES_ROLLOUT` / `NGPUS_PER_NODE` (e.g. 4×8 H20 → `NNODES=4`, `NGPUS_PER_NODE=8`) |
| P2A on/off + strength | `P2A_BONUS_MAP_DIR` (unset = baseline), `P2A_M_MAX` |
| ARL gateway | `ARL_GATEWAY_URL` |
| Hard-subset criterion | `--difficulties` flag of `build_data.py swebench-hard` (default = old rLLM set) |

Hydra training overrides live in `scripts/train_p2a.sh`; if you move any to a json/yaml
config, put it under `config/`.

## ⚠️ TODO — verify before trusting P2A training

The P2A advantage-reshape (`p2a/trainer.py::apply_p2a_reshape` → `p2a/core.py`) is
implemented and wired into the trainer, but it has **never been run end-to-end at
training**. It assumes the Uni-Agent rollout writes each step's `tool_calls` /
`response_text` into the trajectory in the format `apply_p2a_reshape` parses; if it
does not, `reads` is empty and the reshape silently becomes a no-op even with a
bonus map.

**Do a small demo smoke test first** — confirm that P2A actually works on the
Uni-Agent tool set and really captures the agent's actions (non-empty `reads`,
non-trivial multipliers) — **before launching a real training run.**
