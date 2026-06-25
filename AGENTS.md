# AGENTS.md - P2A source repository

This `src/` directory is the source repository for the Uni-Agent migration. It has
a GitHub remote (`origin` = `git@github.com:Siyuexi/TraceAnalyzer.git`); development
is on `main`.

## Layout

- `uni-agent/` - pristine Uni-Agent upstream-mirror submodule. Use this for vanilla baseline code and upstream Uni-Agent docs.
- `env/` - local Uni-Agent environment glue for ARL SDK deployment, image routing, agent-loop adapter, and smoke/data helpers. The external `arl-env` SDK owns the `arl` import name.
- `p2a/` - P2A trainer wrapper, advantage reshape code, and bonus-map precompute utilities.
- `scripts/` - local launch helpers for preparing data/config and running Uni-Agent baseline checks.
- `UNI_AGENT_MIGRATION.md` - current migration notes and tomorrow's baseline commands.

## Asset and artifact paths

- Shared datasets live outside this checkout under `../../datasets` by default.
  Generated P2A parquets use `DATA`, conventionally `../../datasets/p2a`.
- Shared model checkpoints live outside this checkout under `../../models` by
  default. Use `MODEL_PATH`, `MODEL`, or `P2A_MODELS_DIR` to override.
- Project artifacts live inside this checkout under `data/` by default. This
  includes bonus maps, validation details, SQLite eval caches, rollout dumps,
  analysis reports, and dashboard snapshots. Use `P2A_ARTIFACTS_DIR` only when
  the whole artifact root must move.
- Do not put public/reusable datasets or model checkpoints under `src/data`.
  Do not put TraceAnalyzer-specific run artifacts under `../../datasets` or
  `../../models` by default.

## Research concept docs

- The root `proposal.md` is the proposal source of truth, root
  `proposal.html` is the canonical web rendering, and root
  `report/proposal.html` is the VRC-hosted copy. Keep all three synchronized
  when proposal text changes.
- The root `report/2026-06-18_traceanalyzer-logic-map.html` is the living
  implementation/concept map. New P2A concepts should be recorded there with
  stable fully-qualified symbol anchors (`module::function` or
  `module::Class.method`), not brittle file-line references.
- The root `report/2026-06-09_p2a-bonus-map-pipeline.html` is the companion
  reference for bonus-map taxonomy, capture, and classification semantics.
- If a code change introduces a concept that changes research claims, method
  semantics, experiment definitions, or public terminology, update root
  `proposal.md`, `proposal.html`, and `report/proposal.html` in the same unit.

## Semantic consistency rules

- Do not let dashboard-only terminology drift from P2A functionality. Trace
  labels, KPI names, legend text, reports, and README explanations must use the
  same semantics as `p2a/core.py`, `p2a/eval_fault_localization.py`, and
  `p2a/dashboard_adapter.py`.
- Use Graph / Path / Trace terminology consistently. **Graph** means the real
  dependency graph captured from instrumentation/failing-test execution.
  **Path** means the issue symptom-to-root-cause subgraph/path. **Trace** means
  the model/agent execution trajectory. Do not use "Trace" for the captured
  dependency graph in user-facing labels or research text.
- Treat `call_graph_*`, `chain_*`, `dynamic_traceable_*`, and historical
  instrumentation filenames as legacy storage/API vocabulary. New helpers,
  variables, comments, UI labels, README/proposal text, and issue descriptions
  should use Graph, Path, and Trace directly; when old keys are required for
  backward compatibility, isolate them behind explicit alias/normalization
  helpers and label them as legacy.
- If a dashboard feature depends on read/write/error/root-cause semantics, add
  or reuse the corresponding parser/scorer fields in P2A source first, then
  render those fields in the frontend. Avoid frontend-only inference for
  metrics or trace status unless it is a compatibility fallback for old
  artifacts.
- Treat the SQLite eval cache as a raw capture and run-status store by default.
  Dashboard metrics and trace pattern states must be computed from raw rollout
  content plus bonus maps in dashboard/scorer code, not trusted from stale DB
  score fields. New collection paths should store basic facts such as resolved
  state, token usage, runtime, artifacts, and raw rollout content, but should
  not populate localization score columns, `metrics_json.detail`, or pattern
  flags. If dashboard-computed scores are persisted later, the write path is
  one-way dashboard -> DB and the default read path still recomputes.
- Treat node source code as bonus-map data. Dashboard Node Source must read full
  callable source from the inferred or explicit P2A bonus-map directory; DB
  `source_preview` fields are only compatibility fallbacks for old artifacts.
- Execution failure must be inferred from structured tool/runtime signals such
  as `status`, `error`, nonzero exit code, or traceback/command-failure output,
  not from broad keyword scans over source code or successful read observations.
- Keep the unified dashboard compatible with local training, local inference,
  and third-party API inference artifacts. New trace fields should degrade
  cleanly when older artifacts do not contain them.
- When public-facing semantics change, update the README and, where research
  claims are affected, keep the proposal/report documents synchronized.

## ARL is a sandbox, not a VRC remote

ARL is the containerized compute backend. The `arl-env` SDK connects directly to the
ARL Gateway (`ARL_GATEWAY_URL`) to boot a per-instance
sandbox where tests + P2A instrumentation run (bonus-map precompute, training rollouts),
and it is reachable directly from CPU hosts. This has nothing to do with VRC's `remote`
facility: `vrc remote` is the debug proxy that targets the **GPU server** when the local
host has no GPU. An ARL gateway being reachable or not is independent of `vrc remote
health` — do not infer one from the other.

## Dashboard service deployment

- This environment is a server. Do not present dashboard preview URLs bound to
  `127.0.0.1` or `localhost` as user-accessible links.
- For user-accessible dashboard services, bind the server to `0.0.0.0` and give
  the user the server public IP plus port, for example
  `http://<server-public-ip>:8770`.
- When the user says they will manage the service themselves, provide the exact
  start/stop commands only; do not start or keep the service running for them.

## Python Rules

Every Python invocation inside `src/` MUST be prefixed with `uv run` (the repo pins
`uv.lock`). Bare `python`/`pip`/`pytest` either fail (deps not on PATH) or silently
pick a different environment than the lock — `uv run python ...`, `uv run pytest ...`,
`uv run ruff ...` are the only correct forms.

- Keep local P2A imports available with `PYTHONPATH=uni-agent/verl:uni-agent:.` when running from this `src/` directory.
- `UV_CACHE_DIR=/tmp/uv-cache` is a Codex sandbox workaround only: use it when
  Codex cannot write to `~/.cache/uv`, and run through this repo's `uv run` /
  `src/.venv` interpreter rather than another Python. It is not a project
  requirement. Commands written for the user should stay in the normal form
  they can run from `src/`, such as `bash scripts/...` or `uv run ...`, without
  Codex-only cache prefixes.
- For ARL runs, use `scripts/uni_agent_arl.sh`; it keeps `uni-agent/` unmodified and routes runtime startup through `env.agent_loop.ArlUniAgentLoop`.

## Comment Hygiene

Comments describe the present design and the non-obvious WHY — never the code's
history. Do NOT write changelog/defensive comments ("previously did X, it was wrong,
changed to Y", "reverted the … switch", dated attributions, PR/issue numbers as
narrative). Git log is the history; the code is not. Delete such comments or rewrite
them present-tense.

## Uni-Agent / ARL Docs

When working with the Uni-Agent training stack, use the official Uni-Agent
documentation as a primary reference. ARL-specific runtime behavior lives in
this repository under `env/` and `scripts/uni_agent_arl.sh`.
https://uni-agent.readthedocs.io/en/latest/index.html

## Git Rules

- `src/` is a git repo with a GitHub remote (`origin` = `git@github.com:Siyuexi/TraceAnalyzer.git`). Open PRs against `main`; the controller merges. Never self-merge.
- `uni-agent/` is a nested submodule pointing at the pristine fork mirror `git@github.com:Siyuexi/uni-agent.git`.
- Do not modify `uni-agent/`; put P2A behavior in `p2a/`, `env/`, `scripts/`, or `config/`.
- Treat GitHub read access, git-over-SSH push access, and authenticated
  issue/PR comment writes as separate capabilities. Do not infer a valid `gh`
  token from successful GitHub reads or `git push`; before claiming GitHub
  synchronization is blocked, identify the exact operation and access path.
