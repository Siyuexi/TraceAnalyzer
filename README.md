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

## Current capabilities

This repo now has four mostly independent surfaces:

| Surface | Use it for | Primary entry points |
|---|---|---|
| Data setup | Build R2E-Gym subset, SWE-bench Verified, and the default SWE-bench hard validation split. | `scripts/setup.sh data ...`, `scripts/build_data.py` |
| P2A training | Run Uni-Agent baseline or P2A advantage reshaping on ARL/Ray. | `scripts/main.sh`, `scripts/train_p2a.sh` |
| Graph diagnostics | Precompute dynamic/static bonus maps and score whether rollouts read the fault-propagation graph. | `scripts/precompute_eval_bonus_maps.sh`, `p2a.eval_fault_localization`, live `val-p2a/*` validation metrics |
| Third-party baselines | Run an OpenAI-compatible external model through the same Uni-Agent + ARL SWE/R2E environment and produce rollout/localization artifacts. | `scripts/main_3rd.sh`, `scripts/third_party_eval.sh`, `config/third_party_eval.deepseek.example.yaml` |

The shell scripts are layered deliberately: `scripts/setup.sh` owns idempotent
data/dependency/map setup; `main.sh` and `main_3rd.sh` are high-level wrappers;
the remaining scripts are lower-level runners or runtime checks for explicit
control.

## Directory map

```
src/
  config/                     # ALL json/yaml config lives here
    startup_fixups.json       #   per-repo test-startup fixups (source patches + dep pins)
    bad_instances.json        #   R2E instances to exclude from training (skip-list)
    third_party_eval.deepseek.example.yaml # third-party OpenAI-compatible model config template
  scripts/
    build_data.py             # SINGLE data builder: r2e | swebench-verified | swebench-hard | skip-list
    uni_agent_arl.sh          # prepare/data/smoke/debug launcher (ARL config)
    setup.sh                  # idempotent data/dependency/eval-map setup helpers
    ray_setup.sh              # bring up Ray and smoke-check Ray Jobs
    stage_local_runtime.sh    # copy checkout/venv to node-local runtime storage
    main.sh                   # one-shot baseline launcher
    main_3rd.sh               # one-shot third-party OpenAI-compatible rollout baseline
    train_p2a.sh              # training launcher (baseline OR P2A)
    third_party_eval.sh       # lower-level third-party rollout pass-through
    precompute_eval_bonus_maps.sh # eval-map helper for validation diagnostics
    check_deps_cpu.sh         # CPU dependency/import smoke check
    check_uni_agent_runtime.py # GPU/runtime import smoke check
  p2a/
    core.py                   # bonus-map load + read->callgraph match + m(d)=m_max^(1-d) multiplier
    trainer.py                # apply_p2a_reshape: capture agent reads -> reshape advantage
    main.py                   # training entry; P2AFullyAsyncTrainer (vanilla if P2A_BONUS_MAP_DIR unset)
    rollouter.py              # validation rollouter wrapper for live graph diagnostics
    validation_metrics.py     # aggregate val-p2a/* localization metrics
    third_party_eval.py       # OpenAI-compatible external model rollout harness
    test_setup.py             # startup_fixup_command(repo) — loads config/startup_fixups.json
    trace.py                  # instrumentation + call-graph build (bonus-map precompute)
    eval_fault_localization.py # offline eval rollout read->fault-localization metrics
    hf_assets.py              # shared HuggingFace model/dataset path helpers
    runtime_env.py            # Ray runtime-env path normalization helpers
    precompute/precompute_bonus_maps.py  # build bonus maps on the ARL backend
    skip_cases.py             # load_skip_ids() from config/bad_instances.json
  env/                        # ARL deployment + runtime (implements swe-rex AbstractRuntime; no swe-rex server)
  uni-agent/                  # UNMODIFIED Uni-Agent submodule (swe-rex is its runtime interface)
```

## Pipeline

### Step 0. Prepare the shared checkout and venv

Clone the repo onto the shared disk first. Run all following commands from the
repo root:

```bash
git clone git@github.com:Siyuexi/TraceAnalyzer.git
cd TraceAnalyzer
git submodule update --init --recursive
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=$CUDA_HOME
uv python install --managed-python 3.11
"$(uv python find --managed-python --no-project 3.11)" -m venv --clear --copies .venv
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --locked --extra train --extra gpu
```

The `uv` path above is also the fused Megatron / mbridge training runtime.
`scripts/main.sh`, `scripts/train_p2a.sh`, and `scripts/ray_setup.sh` default to
`.venv` and native CUDA 13.0. Verify the runtime before launching a cluster job:

```bash
cd TraceAnalyzer
git submodule update --init --recursive
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=$CUDA_HOME
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --no-sync python scripts/check_uni_agent_runtime.py
```

The locked GPU stack is:

| Component | Version / source |
|---|---|
| CUDA toolkit | `/usr/local/cuda-13.0` |
| PyTorch | CUDA 13 wheels from `https://download.pytorch.org/whl/cu130` |
| vLLM | `vllm==0.11.2` |
| cupy | `cupy-cuda13x==13.6.0` |
| TransformerEngine / Megatron | resolved by `uv sync --locked --extra train --extra gpu` |
| mbridge | `git+https://github.com/ISEEKYAN/mbridge.git` |

`scripts/ray_setup.sh` stages the selected venv to each node-local runtime path,
so head and workers run the same Python stack.

### Step 1. Set common paths

Set these once per shell:

```bash
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export RAY_DATA_HOME="${RAY_DATA_HOME:-$HOME/verl}"
export DATA="${DATA:-../../datasets/p2a}"
export MODEL="${MODEL:-../../models/Qwen3-Coder-30B-A3B-Instruct}"
# Usual ARL Gateway: http://118.145.201.106:80
export ARL_GATEWAY_URL="${ARL_GATEWAY_URL:?set ARL_GATEWAY_URL}"
# If submitting from the Ray head node, the local dashboard endpoint is enough.
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8265}"
# Ray cluster ports. 6379 is Ray GCS; 8265 is Ray dashboard / Jobs.
export RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_SSH_OPTS="${RAY_SSH_OPTS:--p 36000 -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8}"
# Ray 2.55 prestarts one Python worker per advertised CPU. Limit this on
# large GPU nodes so dashboard-agent / Jobs startup is not blocked by hundreds
# of shared-venv worker imports.
export NUM_CPUS="${NUM_CPUS:-64}"
```

### Step 2. Build the training and validation parquets

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

### Step 3. Precompute training bonus maps for P2A

```bash
PYTHONPATH=.:uni-agent:uni-agent/verl P2A_DEPLOYMENT=arl \
  uv run python p2a/precompute/precompute_bonus_maps.py \
    $DATA/r2e_gym_subset_p2a.parquet --mode dynamic --n_parallel 64
```

Training maps default to `../../p2a/bonus_maps`; set `P2A_BONUS_MAP_DIR` to use a
different read/write directory.

Each generated map includes `call_graph_nodes`, `call_graph_edges`, and a
per-node `source` snippet when the sandbox file can be read. The edge list is
diagnostic schema for topology views; training still reshapes from node
distances.

Skip this step for a pure baseline run. P2A training reads these maps through
`P2A_BONUS_MAP_DIR`.

### Step 4. Precompute eval maps for validation graph metrics

```bash
TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  P2A_EVAL_BONUS_MAP_DIR=$DATA/eval_bonus_maps bash scripts/precompute_eval_bonus_maps.sh
```

Eval maps are diagnostic only. They are read during validation logging but never
used by the P2A training reshape.

### Step 5. Configure logging and GPU layout

Configure Weights & Biases before submitting training. Use online logging:

```bash
wandb login
# or export WANDB_API_KEY=...
```

Or use local offline logs:

```bash
export WANDB_MODE=offline
```

The launcher defaults to the Uni-Agent Qwen3 30B MoE 64-GPU shape. For a
4-node x 8-GPU cluster, override it with this 32-GPU starter profile:

```bash
export NNODES_TRAIN=2
export NNODES_ROLLOUT=2
export NGPUS_PER_NODE=8
```

This allocates 16 GPUs to trainer and 16 GPUs to rollout. Do not set
`NNODES_TRAIN=4 NNODES_ROLLOUT=4 NGPUS_PER_NODE=8` on a 32-GPU cluster; that
requests 64 GPUs.

### Step 6. Optional manual Ray restart

`scripts/main.sh` restarts Ray by default before submitting training. To restart
Ray manually instead, run this from the head node. It stops workers first, then
the head, then starts the head, starts workers, and finally submits a tiny Ray
Jobs smoke task:

```bash
HEAD_IP=<HEAD_IP>
export RAY_WORKER_HOSTS="<WORKER_IP_1> <WORKER_IP_2> <WORKER_IP_3>"
export P2A_LOCAL_ROOT=/tmp/p2a-traceanalyzer
export RAY_SSH_OPTS="${RAY_SSH_OPTS:--p 36000 -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8}"
bash scripts/ray_setup.sh "$HEAD_IP" restart-cluster
```

The script keeps the shared checkout as the source of truth, stages repo code,
`.uv-python`, and `.venv` to `$P2A_LOCAL_ROOT/TraceAnalyzer` on each node,
rewrites copied venv paths from the shared source to the local runtime path,
then starts Ray from the local venv. `P2A_STAGE_LOCAL_RUNTIME=1` is the default.
Data, models, checkpoints, and W&B output remain controlled by `DATA`,
`MODEL_PATH`/`MODEL`, `RAY_DATA_HOME`, and `WANDB_DIR`; keep those on shared
storage unless their access pattern becomes a measured bottleneck. The staging
path mainly reduces Ray control-plane / worker-import pressure during startup;
it is not expected to materially speed up steady-state training unless training
was blocked on shared-disk Python package reads.

The script starts `--head` on the head node and joins workers to that head over
ssh. It intentionally ignores the generic `PORT` environment variable; use
`RAY_GCS_PORT` if you need to override the Ray cluster port. `8080` is not a Ray
port in the VRC remote-debug setup. In staged mode, `NUM_CPUS` defaults to
`nproc`; in direct shared-disk mode it defaults to 64 to avoid prestarting
hundreds of workers from the shared venv.

To bypass local staging and run directly from the shared checkout:

```bash
export P2A_STAGE_LOCAL_RUNTIME=0
bash scripts/ray_setup.sh "$HEAD_IP" restart-cluster
```

If you cannot ssh from the head to workers, run the same script manually in this
order:

```bash
HEAD_IP=<HEAD_IP>
# on every worker first
bash scripts/ray_setup.sh "$HEAD_IP" stop
# on the head
bash scripts/ray_setup.sh "$HEAD_IP" start
# on every worker
bash scripts/ray_setup.sh "$HEAD_IP" start
# on the head
bash scripts/ray_setup.sh "$HEAD_IP" smoke
```

The `smoke` command does not restart Ray. It waits for the dashboard and submits
a tiny Ray Jobs task. If you submit training from the head node,
`RAY_API_SERVER_ADDRESS` can stay at `http://127.0.0.1:8265`; otherwise set it
to `http://<HEAD_IP>:8265`.

### Step 7. Submit a baseline or P2A run

One-shot baseline from the Ray head:

```bash
bash scripts/main.sh
```

`scripts/main.sh` stages code/runtime locally like the other launchers, calls
`scripts/setup.sh` for idempotent dependency and data setup, restarts Ray through
`scripts/ray_setup.sh`, and keeps default `DATA` / `MODEL` paths anchored at the
shared checkout (`../../datasets/p2a` and `../../models/...`) instead of under
`/tmp`. It defaults to the current GPU cluster
(`HEAD_IP=28.45.32.245`, `RAY_WORKER_HOSTS="28.45.33.48 28.45.33.95 28.45.33.97"`,
`RAY_GCS_PORT=6379`, `RAY_SSH_OPTS="-p 36000 ..."`). Override those env vars if
the allocation changes. To submit to an already-running Ray cluster without a restart, set
`P2A_RESTART_RAY=0`.

Baseline:

```bash
TRAIN_FILE=$DATA/r2e_gym_subset_p2a.train.parquet \
  TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  MODEL_PATH=$MODEL \
  P2A_EVAL_BONUS_MAP_DIR=$DATA/eval_bonus_maps \
  P2A_EVAL_DETAILS_DIR=$DATA/eval_details \
  bash scripts/train_p2a.sh
```

P2A:

```bash
TRAIN_FILE=$DATA/r2e_gym_subset_p2a.train.parquet \
  TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  MODEL_PATH=$MODEL \
  P2A_EVAL_BONUS_MAP_DIR=$DATA/eval_bonus_maps \
  P2A_EVAL_DETAILS_DIR=$DATA/eval_details \
  P2A_BONUS_MAP_DIR=../../p2a/bonus_maps \
  P2A_M_MAX=3.0 \
  P2A_CREDIT_GRANULARITY=step \
  bash scripts/train_p2a.sh
```

`P2A_CREDIT_GRANULARITY=step` is the default and preserves per-step reshape
behavior. Set `P2A_CREDIT_GRANULARITY=block` to group adjacent same-purpose
steps and apply credit to read blocks that touch the call graph.

### Step 8. Optional offline analysis

For an already-dumped rollout file or dump directory:

```bash
uv run python -m p2a.eval_fault_localization $ROLLOUT_JSONL \
  --bonus-map-dir $DATA/eval_bonus_maps \
  --summary-out $DATA/eval_faultloc_summary.json \
  --details-out $DATA/eval_faultloc_details.jsonl
```

To build the static trace dashboard with summary cards, per-record drill-down,
purpose-block/order/miracle diagnostics, and an expandable graph topology panel:

```bash
uv run python scripts/p2a_dashboard.py $ROLLOUT_JSONL \
  --bonus-map-dir $DATA/eval_bonus_maps \
  --out-dir $DATA/p2a_dashboard
```

Add `--watch --interval 30` when `$ROLLOUT_JSONL` is a live run directory and
you want the same artifacts rebuilt while training writes new dumps.

The offline `summary-out` and `details-out` files are post-hoc artifacts for
inspecting dumped rollouts. Training and validation do not read them; live
validation dashboards use `P2A_EVAL_BONUS_MAP_DIR`.

### Step 9. Optional third-party model rollout baseline

For a main-style smoke/default run, set the API key and run the wrapper. It
defaults to `swebench-hard`, builds the parquet if missing, precomputes matching
dependency/call-graph maps, and writes rollout + fault-localization artifacts
under `$DATA/third_party/<dataset>/<model>/`:

```bash
export P2A_THIRD_PARTY_API_KEY=...
export P2A_THIRD_PARTY_BASE_URL=https://apic1.ohmycdn.com/v1
export P2A_THIRD_PARTY_MODEL=deepseek-v4-flash

bash scripts/main_3rd.sh
```

Switch datasets with `THIRD_PARTY_DATASET=swebench-verified` or
`THIRD_PARTY_DATASET=r2e-gym-subset`. Keep `P2A_THIRD_PARTY_LIMIT` small for
smoke tests; set it higher, or to `all`, for a real baseline. The wrapper does
not sync dependencies by default so it does not prune a shared training `.venv`;
set `P2A_THIRD_PARTY_SYNC_DEPS=1` only when you intentionally want a core CPU
sync in the active environment.

The lower-level pass-through remains available when you want explicit control
over every path and CLI flag:

```bash
export P2A_THIRD_PARTY_API_KEY=...
export P2A_THIRD_PARTY_BASE_URL=https://apic1.ohmycdn.com/v1
export P2A_THIRD_PARTY_MODEL=deepseek-v4-flash

timeout 15m bash scripts/third_party_eval.sh \
  --config config/third_party_eval.deepseek.example.yaml \
  --data $DATA/swe_bench_verified_hard.parquet \
  --out $DATA/third_party/deepseek_v4_flash_rollouts.jsonl \
  --limit 1 \
  --max-turns 3 \
  --max-tokens 1024 \
  --tool-install-timeout 300 \
  --skip-tool-install str_replace_editor \
  --bonus-map-dir $DATA/eval_bonus_maps
```

The harness uses Uni-Agent's `OpenAICompatibleChatModel`, the local ARL
deployment adapter, and the same SWE/R2E reward specs as training. It writes
`p2a_third_party_rollout_v1` JSONL with `messages`, structured tool calls,
`p2a_step_traces`, reward details, and termination status. When
`--bonus-map-dir` is set it also writes scorer details, a summary JSON, and a
short Markdown localization baseline report. For smoke tests, `--max-turns`,
`--max-tokens`, `--tool-install-timeout`, `--skip-tool-install`, the timeout
overrides, and an outer shell `timeout` can bound the ARL/model spend without
editing the checked-in config.

## What you configure yourself

These are knobs you set; the repo does not pin them:

| What | Where |
|---|---|
| Model | `MODEL_PATH` env var; default is `../../models/Qwen3-Coder-30B-A3B-Instruct` from `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Shared generated data root | `DATA`, conventionally `../../datasets/p2a` |
| Train / val data | `TRAIN_FILE` / `TEST_FILE` env vars (point at the parquets built above) |
| Ray job target | `RAY_API_SERVER_ADDRESS` (Ray Jobs endpoint, usually `http://<ray-head-ip>:8265`) |
| GPU / CPU layout | `NNODES_TRAIN` / `NNODES_ROLLOUT` / `NGPUS_PER_NODE` (32-GPU 4×8 starter: `2 / 2 / 8`); `NUM_CPUS` is the Ray CPU resource advertised per node, default 64 |
| Bonus maps (read + write) | `P2A_BONUS_MAP_DIR` — one dir for both precompute output and training input; default `../../p2a/bonus_maps`. Training treats it as the P2A on/off switch (unset = baseline). `P2A_M_MAX` sets strength. `P2A_CREDIT_GRANULARITY=step|block` selects per-step or purpose-block credit. |
| Eval fault-localization diagnostics | `P2A_EVAL_BONUS_MAP_DIR`, `P2A_EVAL_NEAR_THRESHOLD`, `P2A_EVAL_DETAILS_DIR`, `P2A_EVAL_BONUS_N_PARALLEL`, `P2A_EVAL_BONUS_LIMIT`, `P2A_EVAL_BONUS_OFFSET` |
| Third-party provider baseline | `config/third_party_eval.deepseek.example.yaml` plus `P2A_THIRD_PARTY_BASE_URL`, `P2A_THIRD_PARTY_API_KEY`, `P2A_THIRD_PARTY_MODEL`; API keys stay in env vars, not git |
| Third-party run scope | `THIRD_PARTY_DATASET`, `THIRD_PARTY_DATA_FILE`, `P2A_THIRD_PARTY_LIMIT`, `P2A_THIRD_PARTY_N_PARALLEL`, `P2A_THIRD_PARTY_RUN_TIMEOUT`, `P2A_THIRD_PARTY_BONUS_*` |
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
| `avg_node_recall` / `avg_read_precision` / `avg_hit_f1` | Node-level hit recall, read precision, and F1 across scored rollouts. |
| `avg_order_score` / `reverse_order_rate` | Kendall-style agreement between read order and movement from tests toward patched callables. |
| `miracle_rate_over_gt_hits` | Fraction of ground-truth hits that jump directly to patched code before reading intermediate graph levels. |
| `avg_block_order_score` / `block_miracle_rate_over_gt_hits` | Same order and miracle diagnostics after purpose-block segmentation. |
| `block_achieve_rate` / `block_waste_rate` / `block_loop_rate` | Purpose-block outcomes, including repeated same-action loop blocks. |
| `avg_block_efficiency_steps` | Average steps to first call-graph hit inside achieving read blocks. |
| `achieving_block_step_share` / `wasted_block_step_share` / `loop_block_step_share` | Share of block-covered steps spent in each block outcome. |
| `bad_pattern_trace_rate` / `error_spiral_rate` | Trace-level loop and repeated-error flags. |
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
val-p2a/swebench-hard/avg_node_recall
val-p2a/swebench-hard/avg_read_precision
val-p2a/swebench-hard/avg_hit_f1
val-p2a/swebench-hard/order_defined_rate
val-p2a/swebench-hard/reverse_order_rate
val-p2a/swebench-hard/miracle_rate_over_gt_hits
val-p2a/swebench-hard/block_order_defined_rate
val-p2a/swebench-hard/block_reverse_order_rate
val-p2a/swebench-hard/block_miracle_rate_over_gt_hits
val-p2a/swebench-hard/avg_blocks_per_trace
val-p2a/swebench-hard/block_achieve_rate
val-p2a/swebench-hard/block_waste_rate
val-p2a/swebench-hard/block_loop_rate
val-p2a/swebench-hard/achieving_block_step_share
val-p2a/swebench-hard/wasted_block_step_share
val-p2a/swebench-hard/loop_block_step_share
val-p2a/swebench-hard/bad_pattern_trace_rate
val-p2a/swebench-hard/error_spiral_rate
val-p2a/swebench-hard/avg_min_distance_on_hits
val-p2a/swebench-hard/avg_best_positive_multiplier_on_hits
val-p2a/swebench-hard/avg_order_score
val-p2a/swebench-hard/avg_block_order_score
val-p2a/swebench-hard/avg_block_efficiency_steps
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

## Training Smoke Check

Before a full P2A run, do a small smoke run and confirm validation logs include
the `val-p2a/swebench-hard/*` metrics above. If `P2A_EVAL_DETAILS_DIR` is set,
check that it writes per-step JSONL files for the validation cases.
