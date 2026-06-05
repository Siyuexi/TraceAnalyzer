# AGENTS.md - local P2A source repository

This `src/` directory is the local source repository for the Uni-Agent migration.
It intentionally has no remote right now.

## Layout

- `uni-agent/` - Uni-Agent fork submodule. Use this for vanilla baseline code and upstream Uni-Agent docs.
- `env/` - local Uni-Agent environment glue for ARL SDK deployment, image routing, agent-loop adapter, and smoke/data helpers. The external `arl-env` SDK owns the `arl` import name.
- `p2a/` - P2A trainer wrapper, advantage reshape code, and bonus-map precompute utilities.
- `scripts/` - local launch helpers for preparing data/config and running Uni-Agent baseline checks.
- `UNI_AGENT_MIGRATION.md` - current migration notes and tomorrow's baseline commands.

## Python Rules

Do not use the old rLLM `uv run` rule here. Follow Uni-Agent/verl instructions:

- Use `python`, `pip`, and `ray job submit` for Uni-Agent paths.
- Install Uni-Agent from `uni-agent/`.
- Keep local P2A imports available with `PYTHONPATH=uni-agent/verl:uni-agent:.` when running from this `src/` directory.
- For ARL runs, use `scripts/uni_agent_arl.sh`; it keeps `uni-agent/` unmodified and routes runtime startup through `env.agent_loop.ArlUniAgentLoop`.

## Uni-Agent / veFaaS Docs

When working with the Uni-Agent + veFaaS stack, use the official Uni-Agent
documentation as a primary reference:
https://uni-agent.readthedocs.io/en/latest/index.html

## Git Rules

- `src/` itself is a local git repo with no remote.
- `uni-agent/` is a nested submodule pointing at `git@github.com:Siyuexi/uni-agent.git`.
- Do not modify `uni-agent/` unless the controller asks for Uni-Agent fork changes.
