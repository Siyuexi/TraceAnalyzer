# CLAUDE.md — P2A on Uni-Agent + ARL (code-level)

Controller requirements gathered during the 2026-06 migration. Read before
touching this tree. The research-level `CLAUDE.md` is at the repo root.

## Hard rules (controller stated these repeatedly)

1. **Self-contained — never depend on `src-backup`** (code or data). All data is
   sourced from HuggingFace:
   - R2E training: `R2E-Gym/R2E-Gym-Subset` (instances + parsed_commit_content +
     relevant_files). Cases that genuinely fail F2P/P2P on the pair-diag ARL images
     are recorded in `config/bad_instances.json` and excluded from training; the
     `dyyyyyyyy/r2e-gym-subset-filtered` set is not used (its filtering was built for
     a different enterprise registry).
   - SWE-bench eval: `R2E-Gym/SWE-Bench-Verified` (rows + eval fields), with the
     `difficulty` label cross-indexed from `princeton-nlp/SWE-bench_Verified`.
   Old code may be **read for reference only**, never imported/run.
2. **No throwaway / one-off scripts as the pipeline, no post-hoc enrichment.** 不要写冗余代码.
   In particular, **ALL "build data" jobs are subcommands of the single
   `scripts/build_data.py`** (`r2e` / `swebench-verified` / `swebench-hard` / `skip-list`). Do NOT add a
   separate `build_*.py` per dataset — building the hard subset, the skip-list, etc.
   are all "build data", so they go in `build_data.py`. 举一反三.
   **Likewise, EVERY runnable entry/launcher script (`*.sh`, training/data launchers)
   lives in `scripts/`, never in `p2a/`/`env/`** (module code stays there; `p2a/main.py`
   is a `python -m p2a.main` module entry, invoked by `scripts/train_p2a.sh`).
   `train_p2a.sh` was moved `p2a/`→`scripts/` on 2026-06-08.
3. **All json/yaml config goes under `config/`** (`startup_fixups.json`,
   `bad_instances.json`, and any future training config).
4. **Images = pair-diag mirror**, not enterprise. `env/images.py` routes to
   `pair-diag-cn-guangzhou.cr.volces.com/code/...`; the build wrapper writes the
   pair-diag ref into each row. pair-diag is the mirror of the original R2E
   `namanjain12` images and is the reference the old bonus-map report reproduces on.
5. **`uni-agent/` is an UNMODIFIED submodule.** All our glue lives in `p2a/`,
   `env/`, `scripts/`, `config/`. Reuse Uni-Agent's prompt/schema constants by
   import, do not copy. `swe-rex` is Uni-Agent's own runtime interface — required,
   not removable (we implement its `AbstractRuntime` for ARL; we do not run a
   swe-rex server).
6. **Training runtime = uv-managed `.venv` on native CUDA 13.0.** Launchers default
   to `/usr/local/cuda-13.0` and the locked cu130 stack; do not add a parallel
   pip-managed runtime path.

## Asset and artifact paths

- `../../datasets` and `../../models` are the shared roots for reusable datasets
  and model checkpoints. Generated P2A parquets use `DATA`, conventionally
  `../../datasets/p2a`; override the model root with `MODEL_PATH`, `MODEL`, or
  `P2A_MODELS_DIR`.
- `src/data` is the default artifact root for TraceAnalyzer-generated outputs:
  bonus maps, validation details, SQLite eval caches, rollout dumps, analysis
  reports, and dashboard snapshots. Override the root with `P2A_ARTIFACTS_DIR`
  only when needed.
- Keep public datasets and reusable checkpoints out of `src/data`; keep
  project-specific artifacts out of the shared datasets/models roots by default.

## Fixups & skip-list

- `config/startup_fixups.json` = the faithful, behavior-equivalent port of the old
  `test_startup_fixups` (source patches like aiohttp `asyncio.async(`→`create_task(`,
  numpy `is ()`→`== ()`, coveragepy py3.10 opcode, orange3 numpy pin, + dep installs).
  **Do not trim** — removing "redundant-looking" fixups already broke classification
  once. Add new per-repo fixups here; any removal needs a full-gate ablation.
- `config/bad_instances.json` = instances whose F2P/P2P cannot be reproduced even
  with fixups. Excluded from ALL ARL training (RL data, P2A bonus precompute, P2A
  training). Unfixable cases: mark here first; later rebuild correct images and push
  to pair-diag to recover them.

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

## Dashboard service deployment

- This environment is a server. Do not present dashboard preview URLs bound to
  `127.0.0.1` or `localhost` as user-accessible links.
- For user-accessible dashboard services, bind the server to `0.0.0.0` and give
  the user the server public IP plus port, for example
  `http://<server-public-ip>:8770`.
- When the user says they will manage the service themselves, provide the exact
  start/stop commands only; do not start or keep the service running for them.

## P2A advantage — verify before trusting (TODO)

`p2a/trainer.py::apply_p2a_reshape` + `p2a/core.py` implement and wire the reshape
(capture agent reads → match to call graph → `m(d)=m_max^(1-d)` multiplier). It has
**never run end-to-end at training**. Before a real run, do a small demo smoke test
proving P2A works on the Uni-Agent tool set and actually captures actions
(non-empty `reads`). See README "TODO".

## Reminders

- **ARL is the sandbox backend, not a VRC remote.** The `arl-env` SDK connects
  directly to the ARL Gateway (`ARL_GATEWAY_URL`)
  to boot a per-instance container sandbox where tests + P2A instrumentation run
  (bonus-map precompute, training rollouts); it is reachable directly from CPU hosts.
  This is unrelated to VRC's `remote` facility — `vrc remote` targets the **GPU
  server** for command debugging. ARL gateway reachability is independent of
  `vrc remote health`; do not infer one from the other.
- Every Python invocation inside `src/` uses `uv run` (pinned `uv.lock`). Keep
  local P2A imports resolvable with `PYTHONPATH=uni-agent/verl:uni-agent:.` when
  running from this `src/` directory.
- **Comments describe the present design, not the code's history.** Do NOT write
  changelog/defensive comments ("previously did X, it was a bug, changed to Y",
  "reverted the … switch", dated attributions, PR/issue numbers as narrative). Git
  log is the history; the code is not your history book. Keep present-tense WHY
  comments; delete or rewrite the rest.
- Migration validated on a 26-case stratified sample: dynamic signal (standard/direct)
  reproduces the old report 8/8; full case_type ~24/26 (residual = non-dynamic edges).
