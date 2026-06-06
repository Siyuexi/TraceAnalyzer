"""Repo-specific startup fixups + test-setup normalization for R2E reproduction.

Canonical, self-contained fixup adapter. The per-repo fixup commands in
``_startup_fixups.json`` are a **faithful, behavior-equivalent port** of the old
``rllm`` ``utils/p2a/test_startup_fixups.apply_test_startup_fixups`` — generated
by running that exact code's command builders, then baked to data so there is no
runtime dependency on the retired ``src-backup`` tree. Do NOT trim entries: the
migration goal is equivalence, not redesign. Any future removal needs a separate
ablation proving the full gate is unchanged (Claude+Codex, 2026-06-05).

Runs AFTER checkout to the buggy commit and BEFORE P2A instrumentation/test, so
the instrumented + tested source is the fixed-up source. Consumed identically by
the bonus-map precompute, the validation gate, and (via the filtered parquet) the
training paths. uni-agent is never modified.

Fixup families (per repo, faithful to the old pipeline):
  - aiohttp:    ``asyncio.async(`` -> ``asyncio.create_task(``; plugin strip; 13 deps
  - numpy:      ``x is ()`` -> ``x == ()`` (py3.8+ SyntaxError)
  - coveragepy: py3.10 ``dis.opmap`` opcode compat; xdist/flaky strip; mock/
                unittest-mixins/flaky/hypothesis; ModuleCleaner shim; legacy helpers
  - orange3:    pin ``numpy==1.23.5`` + ``scikit-learn==1.1.3``
  - pandas:     strip ``--strict-data-files``; pytest-asyncio; strip unknown asyncio cfg
  - pillow:     expose legacy test helper
  - datalad:    pin ``setuptools<70``

Every applied fixup echoes ``P2A_FIX:<name>`` so the report can attribute causes.
"""

from __future__ import annotations

import json
from pathlib import Path

_FIX = "P2A_FIX:"
_REPO_PATH = "/testbed"

# Faithful per-repo command lists: {repo: [[name, shell_command], ...]}.
# All JSON config lives under src/config/ (this module is in src/p2a/).
_FIXUPS: dict[str, list[list[str]]] = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "startup_fixups.json").read_text(encoding="utf-8")
)


def startup_fixup_command(repo: str) -> str:
    """A single shell snippet applying every startup fixup for ``repo``.

    Each fixup echoes ``P2A_FIX:<name>`` on success; failures are tolerated
    (``|| true``) so one optional dep does not abort the rest. Runs from the repo
    root (``/testbed``) because the source patches use repo-relative paths.
    No-op for repos without registered fixups.
    """
    cmds = _FIXUPS.get((repo or "").lower())
    if not cmds:
        return "true"
    # NEWLINE-separated, never `;`: several fixups are ``python - <<'PY' … PY``
    # heredocs whose terminator MUST sit on its own line. Mark success via the
    # following line's ``$?`` check so the heredoc body stays intact.
    lines = [f"cd {_REPO_PATH} 2>/dev/null"]
    for name, cmd in cmds:
        lines.append(cmd)
        lines.append(f"[ $? -eq 0 ] && echo '{_FIX}{name}' || true")
    return "\n".join(lines)


def parse_fixups(output: str) -> list[str]:
    return [ln[len(_FIX):].strip() for ln in (output or "").splitlines() if ln.startswith(_FIX)]
