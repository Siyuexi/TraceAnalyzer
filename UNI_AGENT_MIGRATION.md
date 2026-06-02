# Uni-Agent Migration Notes

This project now uses a local `src/` source repository for P2A code and Uni-Agent as a nested submodule under `src/uni-agent/`.

## Layout

- `src/`: local P2A source repository. It intentionally has no remote right now.
- `src/p2a/`: P2A advantage reshape, bonus-map loading, and precompute utilities.
- `src/scripts/`: project helper scripts for baseline preparation and launch.
- `src/uni-agent/`: Uni-Agent submodule. Its `origin` is the fork (`git@github.com:Siyuexi/uni-agent.git`), and `upstream` can point to `git@github.com:verl-project/uni-agent.git` when upstream sync is needed.
- `src/uni-agent/verl`: nested verl submodule required by Uni-Agent training scripts.
- `src-backup/`: previous rLLM/TraceAnalyzer submodule. It is preserved as-is and should not be touched for the baseline migration.

## Goal For The Next GPU Run

Run Uni-Agent's baseline path first, without P2A loss changes:

1. Prepare veFaaS runtime env and agent config.
2. Generate R2E-Gym-Subset train parquet. Re-run this after the migration; the
   forked preprocess script now embeds `git checkout <commit_hash>` in each
   sample's `post_setup_cmd` so sandboxes start from the buggy commit rather
   than the fixed image HEAD.
3. Generate SWE-Bench Verified veFaaS eval parquet.
4. Run `examples/agent_train/single_node_debug.sh` from `src/uni-agent/`.

P2A bonus-map instrumentation should be added only after the baseline can run end-to-end.
Lightweight P2A rollout instrumentation is available but disabled by default; set
`UNI_AGENT_P2A_TRACE=1` before `src/scripts/uni_agent_baseline.sh prepare` if you want
per-step spans and parsed tool calls to be carried in rollout `extra_fields`.

Dynamic bonus-map construction now uses Uni-Agent sandboxes by default, not the
old rLLM/ARL backend. The old trace code is only imported as instrumentation and
parser utilities. In dynamic mode, the precompute script starts a Uni-Agent
`AgentEnv`, runs the sample `post_setup_cmd`, explicitly checks out the buggy
commit inferred from `commit_hash` or `instance_id`, instruments `/testbed`, and
runs `/root/run_tests.sh`. This is the path intended for R2E-Gym-Subset veFaaS.

For P2A training, use `src/p2a/train_p2a.sh` from the project root. That script
creates a separate `$RAY_DATA_HOME/data/swe_agent/p2a_runtime_env.yaml` with
`PYTHONPATH=uni-agent/verl:uni-agent:.`, because the P2A package lives in `src/`
while Uni-Agent and verl live under `src/uni-agent/`.

## Required Secrets

Do not commit these. Provide them through the shell environment or by editing the generated runtime env file under `$RAY_DATA_HOME`.

```bash
export VOLCE_ACCESS_KEY="..."
export VOLCE_SECRET_KEY="..."
export VEFAAS_FUNCTION_ID="..."
export VEFAAS_FUNCTION_ROUTE="..."
export VEFAAS_REGION="cn-beijing"
export UNI_AGENT_P2A_TRACE="1"  # optional instrumentation; omit for pure baseline
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
pip install git+https://github.com/R2E-Gym/R2E-Gym.git
```

The exact CUDA/vLLM/SGLang/Megatron dependencies depend on the GPU server and the model recipe. Install those following the Uni-Agent/verl environment already used on that server.
The R2E-Gym package is needed by `examples/data_preprocess/r2e_gym_subset_filtered.py` for `r2egym.commit_models.diff_classes.ParsedCommit`.

## One-Command Project Helper

From the project root:

```bash
src/scripts/uni_agent_baseline.sh prepare
```

This creates:

- `$RAY_DATA_HOME/data/swe_agent/runtime_env.yaml`
- `$RAY_DATA_HOME/data/swe_agent/agent_config.yaml`

If the veFaaS environment variables above are set, the helper writes them into `runtime_env.yaml`. It also lowers the debug rollout config from the upstream high-concurrency defaults.

Generate datasets after Uni-Agent dependencies are installed:

```bash
src/scripts/uni_agent_baseline.sh data
```

Run the baseline debug job after Ray/GPU/model/veFaaS are ready:

```bash
export MODEL_PATH="${RAY_DATA_HOME}/models/Qwen3-4B-Instruct-xml-template"
src/scripts/uni_agent_baseline.sh debug
```

The helper sets:

- `TRAIN_FILE=$RAY_DATA_HOME/data/swe_agent/r2e_gym_subset_filtered.parquet`
- `TEST_FILE=$RAY_DATA_HOME/data/swe_agent/swe_bench_verified_vefaas.parquet`
- `RUNTIME_ENV=$RAY_DATA_HOME/data/swe_agent/runtime_env.yaml`
- `AGENT_CONFIG_PATH=$RAY_DATA_HOME/data/swe_agent/agent_config.yaml`

## Manual Baseline Commands

```bash
cd src/uni-agent
export RAY_DATA_HOME="${RAY_DATA_HOME:-$HOME/verl}"
mkdir -p "${RAY_DATA_HOME}/data/swe_agent"

cp examples/agent_interaction/runtime_env.yaml \
  "${RAY_DATA_HOME}/data/swe_agent/runtime_env.yaml"
cp examples/agent_interaction/agent_config_vefaas.yaml \
  "${RAY_DATA_HOME}/data/swe_agent/agent_config.yaml"

DEPLOYMENT=vefaas python examples/data_preprocess/r2e_gym_subset_filtered.py \
  --local-save-dir "${RAY_DATA_HOME}/data/swe_agent"
DEPLOYMENT=vefaas python examples/data_preprocess/swe_bench_verified.py \
  --local-save-dir "${RAY_DATA_HOME}/data/swe_agent"

export TRAIN_FILE="${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet"
export TEST_FILE="${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_vefaas.parquet"
export RUNTIME_ENV="${RAY_DATA_HOME}/data/swe_agent/runtime_env.yaml"
export AGENT_CONFIG_PATH="${RAY_DATA_HOME}/data/swe_agent/agent_config.yaml"

bash examples/agent_train/single_node_debug.sh
```

## P2A Bonus Map Precompute

After veFaaS credentials and Uni-Agent dependencies are available, dynamic
bonus maps can be generated against the Uni-Agent R2E parquet:

```bash
PYTHONPATH=src \
python -m p2a.precompute.precompute_bonus_maps \
  "${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_filtered.parquet" \
  --output_dir "${RAY_DATA_HOME}/data/swe_agent/bonus_maps" \
  --mode dynamic \
  --sandbox_backend uni_agent \
  --n_parallel 4 \
  --limit 20 \
  --save_trace_sidecars
```

Use low `--n_parallel` first because each dynamic item starts a sandbox. The
`legacy` sandbox backend exists only as an explicit fallback:

```bash
P2A_SANDBOX_BACKEND=legacy python -m p2a.precompute.precompute_bonus_maps ...
```

## Known Tomorrow Blockers

- GPU dependencies and Ray cluster are not configured in this workspace yet.
- veFaaS credentials/function/route are not configured yet.
- `single_node_debug.sh` submits a Ray job; Ray must be running on the GPU server.
- Uni-Agent's R2E-Gym-Subset preprocess currently supports veFaaS mapping, not local deployment.
