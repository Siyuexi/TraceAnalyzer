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

## ARL is a sandbox, not a VRC remote

ARL is the containerized compute backend. The `arl-env` SDK connects directly to the
ARL Gateway (`ARL_GATEWAY_URL`, e.g. `http://118.145.201.106:80`) to boot a per-instance
sandbox where tests + P2A instrumentation run (bonus-map precompute, training rollouts),
and it is reachable directly from CPU hosts. This has nothing to do with VRC's `remote`
facility: `vrc remote` is the debug proxy that targets the **GPU server** when the local
host has no GPU. An ARL gateway being reachable or not is independent of `vrc remote
health` — do not infer one from the other.

## Python Rules

Every Python invocation inside `src/` MUST be prefixed with `uv run` (the repo pins
`uv.lock`). Bare `python`/`pip`/`pytest` either fail (deps not on PATH) or silently
pick a different environment than the lock — `uv run python ...`, `uv run pytest ...`,
`uv run ruff ...` are the only correct forms.

- Keep local P2A imports available with `PYTHONPATH=uni-agent/verl:uni-agent:.` when running from this `src/` directory.
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
