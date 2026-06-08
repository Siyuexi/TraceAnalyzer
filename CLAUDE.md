# CLAUDE.md — P2A on Uni-Agent + ARL (code-level)

Controller requirements gathered during the 2026-06 migration. Read before
touching this tree. The research-level `CLAUDE.md` is at the repo root.

## Hard rules (controller stated these repeatedly)

1. **Self-contained — never depend on `src-backup`** (code or data). All data is
   sourced from HuggingFace:
   - R2E training: `dyyyyyyyy/r2e-gym-subset-filtered` (instances + parsed_commit_content)
     joined with `R2E-Gym/R2E-Gym-Subset` for `relevant_files`.
   - SWE-bench validation: `princeton-nlp/SWE-bench_Verified` (has `difficulty`).
   Old code may be **read for reference only**, never imported/run.
2. **No throwaway / one-off scripts as the pipeline, no post-hoc enrichment.** 不要写冗余代码.
   In particular, **ALL "build data" jobs are subcommands of the single
   `scripts/build_data.py`** (`r2e` / `swebench-hard` / `skip-list`). Do NOT add a
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

## P2A advantage — verify before trusting (TODO)

`p2a/trainer.py::apply_p2a_reshape` + `p2a/core.py` implement and wire the reshape
(capture agent reads → match to call graph → `m(d)=m_max^(1-d)` multiplier). It has
**never run end-to-end at training**. Before a real run, do a small demo smoke test
proving P2A works on the Uni-Agent tool set and actually captures actions
(non-empty `reads`). See README "TODO".

## Reminders

- Every Python invocation inside `src/` uses `uv run` (pinned `uv.lock`).
- **Comments describe the present design, not the code's history.** Do NOT write
  changelog/defensive comments ("previously did X, it was a bug, changed to Y",
  "reverted the … switch", dated attributions, PR/issue numbers as narrative). Git
  log is the history; the code is not your history book. Keep present-tense WHY
  comments; delete or rewrite the rest.
- Migration validated on a 26-case stratified sample: dynamic signal (standard/direct)
  reproduces the old report 8/8; full case_type ~24/26 (residual = non-dynamic edges).
