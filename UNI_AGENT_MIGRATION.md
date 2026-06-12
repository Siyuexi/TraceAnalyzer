# Uni-Agent Migration Notes

This project now uses a local `src/` source repository for P2A code and Uni-Agent as a nested submodule under `src/uni-agent/`.

## Layout

- `src/`: P2A source repository with GitHub remote
  `git@github.com:Siyuexi/TraceAnalyzer.git`; development is on `main`.
- `src/p2a/`: P2A advantage reshape, bonus-map loading, and precompute utilities.
- `src/env/`: local ARL SDK deployment glue, image routing, ARL-aware Uni-Agent loop adapter, and smoke/data helpers. The external `arl-env` SDK owns the `arl` import name.
- `src/scripts/`: project helper scripts for baseline preparation and launch.
- `src/uni-agent/`: Uni-Agent submodule. Its `origin` is the fork (`git@github.com:Siyuexi/uni-agent.git`), and `upstream` can point to `git@github.com:verl-project/uni-agent.git` when upstream sync is needed.
- `src/uni-agent/verl`: nested verl submodule required by Uni-Agent training scripts.
- `src-backup/`: previous rLLM/TraceAnalyzer submodule. It is preserved as-is and should not be touched for the baseline migration.

## Baseline Reproduction Path

Run the current ARL baseline path first, without P2A advantage reshape:

1. Prepare ARL runtime env and agent config.
2. Generate R2E-Gym-Subset train parquet. Re-run this after the migration; the
   local builder embeds `git checkout <commit_hash>` in each sample's
   `post_setup_cmd` so sandboxes start from the buggy commit rather than the
   fixed image HEAD.
3. Generate SWE-Bench Verified HARD eval parquet.
4. Run `src/scripts/train_p2a.sh` with `P2A_BONUS_MAP_DIR` unset.

P2A bonus-map instrumentation should be added only after the baseline can run end-to-end.
Lightweight P2A rollout instrumentation is available but disabled by default; set
`UNI_AGENT_P2A_TRACE=1` before `src/scripts/uni_agent_arl.sh prepare` if you want
per-step spans and parsed tool calls to be carried in rollout `extra_fields`.

Dynamic bonus-map construction uses Uni-Agent sandboxes (ARL backend). The trace
instrumentation/parsing engine is a first-class module of this source tree,
`p2a/trace.py` (no dependency on `src-backup`). In dynamic mode, the precompute script starts a Uni-Agent
`AgentEnv`, runs the sample `post_setup_cmd`, explicitly checks out the buggy
commit inferred from `commit_hash` or `instance_id`, instruments `/testbed`, and
runs `/root/run_tests.sh`. This is the path intended for the R2E-Gym-Subset
ARL/Uni-Agent dynamic bonus-map build.

For P2A training, use `src/scripts/train_p2a.sh` from the project root. That script
creates or updates `$RAY_DATA_HOME/data/swe_agent/runtime_env_arl.yaml` with
`PYTHONPATH=uni-agent/verl:uni-agent:.`, and uses
`$RAY_DATA_HOME/data/swe_agent/agent_config_arl.yaml` by default. Leave
`P2A_BONUS_MAP_DIR` unset to reproduce the Uni-Agent baseline; set it only for
P2A advantage reshape.

## Required ARL Environment

Do not commit credentials or cluster-local endpoints. Provide them through the shell
environment or by editing the generated runtime env file under `$RAY_DATA_HOME`.

```bash
export ARL_GATEWAY_URL="http://118.145.210.10:8080"  # override if your ARL gateway differs
export ARL_NAMESPACE="default"
export ARL_EXPERIMENT_ID="p2a-uniagent-arl"
export UNI_AGENT_P2A_TRACE="1"  # optional process tracing; omit for pure baseline
```

## Uni-Agent Install Commands

Run these on the GPU server after entering the Python environment you want to use:

```bash
cd src
git submodule update --init --recursive
cd uni-agent
pip install --no-deps -e ./verl
pip install -e .
pip install swe-rex loguru pydantic pydantic_settings aiohttp datasets ray orjson
pip install arl-env==0.3.1  # required by the direct ARL SDK deployment path
pip install git+https://github.com/R2E-Gym/R2E-Gym.git
```

The exact CUDA/vLLM/SGLang/Megatron dependencies depend on the GPU server and the model recipe. Install those following the Uni-Agent/verl environment already used on that server.
The R2E-Gym package is needed by `examples/data_preprocess/r2e_gym_subset_filtered.py` for `r2egym.commit_models.diff_classes.ParsedCommit`.

For the fused Megatron/mbridge baseline, the current target is the Uni-Agent
cu128 runtime rather than the generic cu130 uv environment:

```bash
cd src
export P2A_CU128_CUDA_HOME=/usr/local/cuda-12.8
bash scripts/setup_uni_agent_cu128_runtime.sh
source .venv-cu128/p2a-cu128.env
python scripts/check_uni_agent_runtime.py
```

The setup script follows `uni-agent/verl/docker/verl0.6-cu128-torch2.8.0-fa2.7.4`:
torch 2.8.0/cu128, flash-attn 2.7.4.post1, TransformerEngine v2.2.1,
Megatron-LM core_v0.13.0, and mbridge. `scripts/main.sh`,
`scripts/train_p2a.sh`, and `scripts/ray_setup.sh` prefer `.venv-cu128` when it
exists and stage that selected runtime to worker-local disk. `.venv` remains the
uv-managed environment; do not run `uv sync` into `.venv-cu128`.

## One-Command Project Helper

From the project root:

```bash
src/scripts/uni_agent_arl.sh prepare
```

This creates:

- `$RAY_DATA_HOME/data/swe_agent/runtime_env_arl.yaml`
- `$RAY_DATA_HOME/data/swe_agent/agent_config_arl.yaml`

If ARL/P2A environment variables above are set, the helper writes them into
`runtime_env_arl.yaml`. It also lowers the debug rollout config from the
upstream high-concurrency defaults when running through `uni_agent_arl.sh debug`.

Generate datasets after Uni-Agent dependencies are installed:

```bash
src/scripts/uni_agent_arl.sh data
```

Run the baseline debug job after Ray/GPU/model/ARL are ready:

```bash
src/scripts/uni_agent_arl.sh smoke
src/scripts/uni_agent_arl.sh debug
```

The helper sets:

- `TRAIN_FILE=$RAY_DATA_HOME/data/swe_agent/r2e_gym_subset_p2a.train.parquet`
- `TEST_FILE=$RAY_DATA_HOME/data/swe_agent/r2e_gym_subset_p2a.train.parquet` for debug
- `RUNTIME_ENV=$RAY_DATA_HOME/data/swe_agent/runtime_env_arl.yaml`
- `AGENT_CONFIG_PATH=$RAY_DATA_HOME/data/swe_agent/agent_config_arl.yaml`

## Manual Baseline Commands

```bash
cd src
export RAY_DATA_HOME="${RAY_DATA_HOME:-$HOME/verl}"
mkdir -p "${RAY_DATA_HOME}/data/swe_agent"

cp uni-agent/examples/agent_interaction/runtime_env.yaml \
  "${RAY_DATA_HOME}/data/swe_agent/runtime_env_arl.yaml"
cp env/agent_config_arl.yaml \
  "${RAY_DATA_HOME}/data/swe_agent/agent_config_arl.yaml"

PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py r2e \
    --out "${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_p2a.parquet"
PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py swebench-hard \
    --out "${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_hard.parquet"

export TRAIN_FILE="${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_p2a.train.parquet"
export TEST_FILE="${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_hard.parquet"
export RUNTIME_ENV="${RAY_DATA_HOME}/data/swe_agent/runtime_env_arl.yaml"
export AGENT_CONFIG_PATH="${RAY_DATA_HOME}/data/swe_agent/agent_config_arl.yaml"

# Baseline: leave P2A_BONUS_MAP_DIR unset.
bash scripts/train_p2a.sh
```

## P2A Bonus Map Precompute

After ARL connectivity and Uni-Agent dependencies are available, dynamic
bonus maps can be generated against the Uni-Agent R2E parquet:

```bash
PYTHONPATH=src \
python -m p2a.precompute.precompute_bonus_maps \
  "${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_p2a.train.parquet" \
  --output_dir "${RAY_DATA_HOME}/data/swe_agent/bonus_maps" \
  --mode dynamic \
  --sandbox_backend uni_agent \
  --n_parallel 4 \
  --limit 20 \
  --save_trace_sidecars
```

Use low `--n_parallel` first because each dynamic item starts a sandbox. The
only sandbox backend is `uni_agent` (backed by the ARL deployment).

The Uni-Agent dynamic path applies the sample `post_setup_cmd`, performs a
plain buggy checkout, instruments `/testbed`, and runs `/root/run_tests.sh`.
There is intentionally no generalized startup-fixup layer in the current ARL
migration path.

## Known Tomorrow Blockers

- GPU dependencies and Ray cluster are not configured in this workspace yet.
- `train_p2a.sh` submits a Ray job; Ray must be running on the GPU server.
- The direct ARL SDK path requires `arl-env==0.3.1` in the ambient Uni-Agent
  execution environment and in the training image/environment; this source tree
  currently does not own a `src/pyproject.toml`/`uv.lock`.

## ARL-backed Uni-Agent Path

ARL is integrated without editing the `uni-agent/` submodule. The local
`env.agent_loop.ArlUniAgentLoop` intercepts `env.deployment.type=arl`, boots an
ARL managed sandbox through the external `arl-env` SDK, and hands Uni-Agent a
local runtime adapter that implements the SWE-ReX `AbstractRuntime` interface
over ARL's persistent interactive shell.

Key commands from `src/`:

```bash
scripts/uni_agent_arl.sh prepare
scripts/uni_agent_arl.sh smoke
scripts/uni_agent_arl.sh data
```

`smoke` is the hard gate before training. It verifies that the ARL SDK sandbox
is reachable, that `run_in_session` persists `export`/`cd` through the
interactive shell, and that `upload` works. If this fails, do not start
training; the ARL runtime adapter needs to be fixed first.

R2E image routing defaults:

- `coveragepy` and `orange3` use the verified enterprise image
  `enterprise-public-cn-beijing.cr.volces.com/r2e-gym-subset/{instance}:latest`.
- Other `namanjain12/*_final` images are rewritten through the ARL mirror
  `${ARL_MIRROR_REGISTRY:-pair-diag-cn-guangzhou.cr.volces.com}/${ARL_MIRROR_NAMESPACE:-code}/...`.
- Extend `P2A_ARL_ENTERPRISE_REPOS` or set exact `P2A_ARL_IMAGE_OVERRIDES_JSON`
  after the full reproduction-gate audit identifies more anomalous images.

For dynamic bonus-map precompute, set `P2A_DEPLOYMENT=arl` while keeping
`--sandbox_backend uni_agent`; the existing adapter will use the ARL bridge and
still run the plain buggy checkout gate before instrumentation.
