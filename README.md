# P2A on Uni-Agent + ARL

Program-Analysis-based Process Advantage (P2A) for SWE agentic RL, implemented on
the **Uni-Agent** training stack with our **ARL** cluster as the sandbox backend.
P2A reshapes the per-step RL advantage using a precomputed **bonus map** (the
golden runtime call graph from failing test → patched callable): steps whose agent
actions land on the fault-propagation path get a larger advantage.

Everything is **self-contained**: data comes from HuggingFace, images from the
pair-diag mirror of the original R2E images. There is **no dependency on the old
`src-backup` fork**.

> **ARL is the sandbox, not a "remote".** The `arl-env` SDK connects directly to the
> ARL Gateway (`ARL_GATEWAY_URL`) to boot a per-instance container sandbox where tests
> and P2A instrumentation run (bonus-map precompute, training rollouts); it is reachable
> directly from CPU hosts. This is separate from VRC's `remote` facility, which targets
> the **GPU server** for command debugging — ARL gateway reachability is independent of
> `vrc remote`.

Default HuggingFace assets are shared across sibling projects:

| Asset | Default location from `src/` | Override |
|---|---|---|
| Models | `../../models/<repo-name>` | `MODEL_PATH` / `P2A_MODELS_DIR` / `P2A_MODEL_REPO` |
| Datasets | `../../datasets/<repo-name>/<split>` | `P2A_DATASETS_DIR` |
| Generated P2A data | `../../datasets/p2a` | `DATA` / `P2A_SHARED_ROOT` |

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
    build_data.py             # SINGLE data builder: r2e | swebench-verified | swebench-hard | skip-list
    uni_agent_arl.sh          # prepare/data/smoke/debug launcher (ARL config)
    ray_setup.sh              # bring up Ray
    train_p2a.sh              # training launcher (baseline OR P2A)
  p2a/
    core.py                   # bonus-map load + read->callgraph match + m(d)=m_max^(1-d) multiplier
    trainer.py                # apply_p2a_reshape: capture agent reads -> reshape advantage   [see TODO]
    main.py                   # training entry; P2AFullyAsyncTrainer (vanilla if P2A_BONUS_MAP_DIR unset)
    test_setup.py             # startup_fixup_command(repo) — loads config/startup_fixups.json
    trace.py                  # instrumentation + call-graph build (bonus-map precompute)
    eval_fault_localization.py # offline eval rollout read->fault-localization metrics
    precompute/precompute_bonus_maps.py  # build bonus maps on the ARL backend
    skip_cases.py             # load_skip_ids() from config/bad_instances.json
  env/                        # ARL deployment + runtime (implements swe-rex AbstractRuntime; no swe-rex server)
  uni-agent/                  # UNMODIFIED Uni-Agent submodule (swe-rex is its runtime interface)
```

## Pipeline

Set the shared paths once per shell:

```bash
export DATA="${DATA:-../../datasets/p2a}"
export MODEL="${MODEL:-../../models/Qwen3-Coder-30B-A3B-Instruct}"
export ARL_GATEWAY_URL="${ARL_GATEWAY_URL:?set ARL_GATEWAY_URL}"
```

1. Build the R2E training parquet and SWE-bench Verified validation parquets:

```bash
PYTHONPATH=.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py r2e --out $DATA/r2e_gym_subset_p2a.parquet

PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py swebench-verified --out $DATA/swe_bench_verified.parquet

PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py swebench-hard --out $DATA/swe_bench_verified_hard.parquet
```

`swebench-hard` is a filtered parquet, not a separate HuggingFace cache directory.
The two upstream cache directories are expected: `SWE-Bench-Verified/` carries
R2E-Gym eval rows, while `SWE-bench_Verified/` carries Princeton difficulty labels.

2. Precompute training bonus maps on ARL:

```bash
PYTHONPATH=.:uni-agent:uni-agent/verl P2A_DEPLOYMENT=arl \
  uv run python p2a/precompute/precompute_bonus_maps.py \
    $DATA/r2e_gym_subset_p2a.parquet --mode dynamic --n_parallel 64
```

Training maps default to `../../p2a/bonus_maps`; set `P2A_BONUS_MAP_DIR` to use a
different read/write directory.

3. Precompute eval maps for live validation graph metrics:

```bash
TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  P2A_EVAL_BONUS_MAP_DIR=$DATA/eval_bonus_maps bash scripts/precompute_eval_bonus_maps.sh
```

Eval maps are diagnostic only. They are read during validation logging but never
used by the P2A training reshape.

4. Launch a baseline run:

```bash
TRAIN_FILE=$DATA/r2e_gym_subset_p2a.train.parquet \
  TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  MODEL_PATH=$MODEL \
  P2A_EVAL_BONUS_MAP_DIR=$DATA/eval_bonus_maps \
  P2A_EVAL_DETAILS_DIR=$DATA/eval_details \
  bash scripts/train_p2a.sh
```

5. Launch a P2A run:

```bash
TRAIN_FILE=$DATA/r2e_gym_subset_p2a.train.parquet \
  TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  MODEL_PATH=$MODEL \
  P2A_EVAL_BONUS_MAP_DIR=$DATA/eval_bonus_maps \
  P2A_EVAL_DETAILS_DIR=$DATA/eval_details \
  P2A_BONUS_MAP_DIR=../../p2a/bonus_maps \
  P2A_M_MAX=3.0 \
  bash scripts/train_p2a.sh
```

Optional offline analysis for an already-dumped rollout file:

```bash
uv run python -m p2a.eval_fault_localization $ROLLOUT_JSONL \
  --bonus-map-dir $DATA/eval_bonus_maps \
  --summary-out $DATA/eval_faultloc_summary.json \
  --details-out $DATA/eval_faultloc_details.jsonl
```

The offline `summary-out` and `details-out` files are post-hoc artifacts for
inspecting dumped rollouts. Training and validation do not read them; live
validation dashboards use `P2A_EVAL_BONUS_MAP_DIR`.

## What you configure yourself

These are knobs you set; the repo does not pin them:

| What | Where |
|---|---|
| Model | `MODEL_PATH` env var; default is `../../models/Qwen3-Coder-30B-A3B-Instruct` from `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Shared generated data root | `DATA`, conventionally `../../datasets/p2a` |
| Train / val data | `TRAIN_FILE` / `TEST_FILE` env vars (point at the parquets built above) |
| GPU layout | `NNODES_TRAIN` / `NNODES_ROLLOUT` / `NGPUS_PER_NODE` (e.g. 4×8 H20 → `NNODES=4`, `NGPUS_PER_NODE=8`) |
| Bonus maps (read + write) | `P2A_BONUS_MAP_DIR` — one dir for both precompute output and training input; default `../../p2a/bonus_maps`. Training treats it as the P2A on/off switch (unset = baseline). `P2A_M_MAX` sets strength. |
| Eval fault-localization diagnostics | `P2A_EVAL_BONUS_MAP_DIR`, `P2A_EVAL_NEAR_THRESHOLD`, `P2A_EVAL_DETAILS_DIR`, `P2A_EVAL_BONUS_N_PARALLEL`, `P2A_EVAL_BONUS_LIMIT`, `P2A_EVAL_BONUS_OFFSET` |
| ARL gateway | `ARL_GATEWAY_URL` |
| Hard-subset criterion | `--difficulties` flag of `build_data.py swebench-hard` (default = the `1-4 hours` / `>4 hours` difficulty set) |
| R2E bad-case policy | `config/bad_instances.json` (pair-diag ARL gate evidence) |

Hydra training overrides live in `scripts/train_p2a.sh`; if you move any to a json/yaml
config, put it under `config/`.

## Eval Fault-Localization Metrics

`scripts/precompute_eval_bonus_maps.sh` reuses the same dynamic precompute path as
training bonus maps, but points it at `TEST_FILE` / `EVAL_FILE`.  The resulting
eval maps should stay out of `P2A_BONUS_MAP_DIR`; they are only a diagnostic
reference for validation rollouts.

`p2a.eval_fault_localization` accepts rollout dumps in `.jsonl`, `.json`, or
`.parquet` format.  It first reads `p2a_step_traces`, then structured
`tool_calls`, then response text / assistant messages, and reports:

| Metric | Meaning |
|---|---|
| `bonus_map_coverage` | Fraction of rollout rows with a matching eval bonus map. |
| `call_graph_coverage` | Fraction with a bonus map that contains call-graph nodes. |
| `read_rate` | Fraction of rows where file-viewing actions were recovered. |
| `graph_hit_rate_over_call_graphs` | Fraction whose reads hit any node in the eval call graph. |
| `ground_truth_hit_rate_over_call_graphs` | Fraction whose reads hit a patched callable (`distance == 0`). |
| `near_hit_rate_over_call_graphs` | Fraction whose best read distance is `<= --near-threshold` (default `0.5`). |
| `avg_min_distance_on_hits` | Lower is better; `0` means the model read the edited callable. |
| `avg_best_positive_multiplier_on_hits` | The diagnostic P2A multiplier implied by the best read distance. |

For live training dashboards, set `P2A_EVAL_BONUS_MAP_DIR` when launching
`scripts/train_p2a.sh`. The local `P2AFullyAsyncRollouter` is the dashboard
wrapper: it keeps the validation path otherwise unchanged, scores validation
rollouts against those eval maps, and returns the same aggregate signals to the
trainer logger at each validation step. For the hard split built by this repo,
the W&B/console keys are:

```
val-p2a/swebench-hard/bonus_map_coverage
val-p2a/swebench-hard/call_graph_coverage
val-p2a/swebench-hard/read_rate
val-p2a/swebench-hard/graph_hit_rate_over_call_graphs
val-p2a/swebench-hard/ground_truth_hit_rate_over_call_graphs
val-p2a/swebench-hard/near_hit_rate_over_call_graphs
val-p2a/swebench-hard/avg_min_distance_on_hits
val-p2a/swebench-hard/avg_best_positive_multiplier_on_hits
```

`P2A_EVAL_DETAILS_DIR` optionally writes per-case JSONL files named by validation
step for debugging individual instances. Those files are an auxiliary dump; the
dashboard metrics above are returned directly from validation and do not depend
on `summary-out` / `details-out` from the offline CLI.

Current SWE-bench Verified eval-map sanity check, after the targeted F2P,
trace-capture, unittest-description F2P, zero-test runner, and F2P collection
guards, is:

| Split | Rows | Dynamic (`standard+direct`) | `standard` | `direct` | `newly_created` | `no_callable` | `no_f2p` | `instrumentation_failed` | `signature_mismatch` | `all_pass` | `no_trace` | `no_gt` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard validation | 45 | 39 (86.7%) | 32 | 7 | 4 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| test rest | 455 | 389 (85.5%) | 258 | 131 | 35 | 18 | 2 | 2 | 7 | 2 | 0 | 0 |
| full Verified | 500 | 428 (85.6%) | 290 | 138 | 39 | 18 | 3 | 3 | 7 | 2 | 0 | 0 |

The full run used `cache/eval_bonus_verified500_f2p_targeted_20260608_220312/bonus_maps/`;
the 13 former `no_f2p` Django cases were rerun under
`cache/swe_no_f2p_rerun_20260609/maps/`, recovering 11 dynamic maps.
The 13 former `all_pass` cases were rerun after the issue #6/#7/#8 merges under
`cache/swe_allpass_after_merge_20260609_131500/maps/`, recovering 10 dynamic
maps. The remaining residuals are 2 deterministic `all_pass` Django cases and 1
`no_f2p` SymPy case.
`no_trace=0` is the build-quality gate.  Non-dynamic buckets now mean:

| Bucket | Count | Meaning |
|---|---:|---|
| `newly_created` | 39 | The patched callable is absent in the buggy tree, so a function-body tracer cannot observe it dynamically. |
| `no_callable` | 18 | The patch has no callable-level Python change for the bonus-map extractor. |
| `signature_mismatch` | 7 | The F2P test fails during Python argument binding before entering the patched callable body. |
| `instrumentation_failed` | 3 | Static callable extraction found candidates, but sandbox instrumentation produced no instrumented callable. |
| `no_f2p` | 3 | F2P failures remain unaligned with traces after description-to-method recovery. |
| `all_pass` | 2 | Buggy F2P tests exit 0 after checkout/test-selection verification, so the bug does not reproduce locally. |

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
