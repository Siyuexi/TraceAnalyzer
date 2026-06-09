"""Self-contained data builder for P2A on Uni-Agent — ONE entry, no old code.

Every "build data" job lives here as a subcommand:
  r2e            HF -> R2E training parquet (full + skip-filtered .train.parquet)
  swebench-verified HF -> SWE-bench-Verified full eval set
  swebench-hard     HF -> SWE-bench-Verified HARD subset (validation)
  skip-list      regenerate config/bad_instances.json from gate results (maintenance)

Each sources from HuggingFace and reuses Uni-Agent's schema/prompt constants by
import (never copied). No dependency on the retired src-backup fork.

Usage (from src/, HF reachable):
  PYTHONPATH=.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess \
    uv run python scripts/build_data.py r2e          --out <path>/r2e_gym_subset_p2a.parquet
    uv run python scripts/build_data.py swebench-verified --out <path>/swe_bench_verified.parquet
    uv run python scripts/build_data.py swebench-hard     --out <path>/swe_bench_verified_hard.parquet
    uv run python scripts/build_data.py skip-list     --gate <gate.jsonl> [--gate ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import pandas as pd

# data_preprocess example modules require a known DEPLOYMENT value at import; we use
# only their prompt constants. This is not the runtime backend. The parquet rows
# below carry pair-diag image refs, and agent_config_arl.yaml supplies ARL at launch.
os.environ.setdefault("DEPLOYMENT", "vefaas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path

MIRROR = "pair-diag-cn-guangzhou.cr.volces.com/code"
HARD_DIFFICULTIES = {"1-4 hours", ">4 hours"}
CONFIG = Path(__file__).resolve().parents[1] / "config" / "bad_instances.json"


# ── r2e ───────────────────────────────────────────────────────────────────────
def cmd_r2e(args) -> int:
    from p2a.hf_assets import load_shared_dataset
    from p2a.skip_cases import load_skip_ids
    from r2egym.commit_models.diff_classes import ParsedCommit
    # Import Uni-Agent's prompt/setup constants only.  The row source below is
    # the canonical R2E dataset, not dyyyyyyyy/r2e-gym-subset-filtered.
    from r2e_gym_subset_filtered import POST_SETUP_CMD, SYSTEM_PROMPT, USER_PROMPT

    def ident(repo, pc_json, commit):
        pc = ParsedCommit(**json.loads(pc_json))
        fixed = pc.new_commit_hash or commit
        buggy = pc.old_commit_hash or f"{fixed}^"
        return f"{repo}__{fixed[:10]}", fixed, buggy

    print("Loading R2E-Gym/R2E-Gym-Subset ...", flush=True)
    rows, rel_hit, rel_miss = [], 0, 0
    for ex in load_shared_dataset("R2E-Gym/R2E-Gym-Subset", split="train"):
        repo, pc_json = ex["repo_name"], ex["parsed_commit_content"]
        iid, fixed, buggy = ident(repo, pc_json, ex["commit_hash"])
        md = {
            "repo": repo, "instance_id": iid, "commit_hash": buggy, "old_commit_hash": buggy,
            "new_commit_hash": fixed, "patch": ParsedCommit(**json.loads(pc_json)).get_patch(),
            "expected_output_json": ex["expected_output_json"],
        }
        relevant_files = ex.get("relevant_files")
        rel = json.dumps(list(relevant_files)) if relevant_files is not None else None
        rel_hit += rel is not None
        rel_miss += rel is None
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(problem_statement=ex["problem_statement"])},
            ],
            "agent_name": "swe_agent",
            "extra_info": {"tools_kwargs": {
                "env": {"deployment": {"image": f"{MIRROR}/{repo}_final:{fixed}"},
                        "post_setup_cmd": POST_SETUP_CMD.format(base_commit=shlex.quote(buggy))},
                "reward": {"name": "r2e_gym", "metadata": md},
            }},
            "parsed_commit_content": pc_json,   # flat top-level (avoids nested-chunk read error)
            "relevant_files": rel,              # JSON string; normalize_task decodes it
        })
    if rel_miss:
        print(f"WARNING: {rel_miss} rows had no relevant_files (static layer widens)", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows (relevant_files {rel_hit} hit / {rel_miss} miss) -> {args.out}", flush=True)

    if not args.no_skip_filter:
        skip = load_skip_ids()
        ids = df["extra_info"].map(lambda ei: ei["tools_kwargs"]["reward"]["metadata"]["instance_id"])
        kept = df[~ids.isin(skip)]
        train_out = args.train_out or f"{args.out[:-len('.parquet')]}.train.parquet"
        kept.to_parquet(train_out, index=False)
        print(f"wrote {len(kept)} rows (skip-list excluded {len(df) - len(kept)}/{len(skip)}) -> {train_out}", flush=True)
    return 0


# ── swebench-verified / swebench-hard ─────────────────────────────────────────
def cmd_swebench(args) -> int:
    from p2a.hf_assets import load_shared_dataset
    # swe_bench_verified's modal branch is import-safe on Python <3.11; we use only its prompts.
    os.environ["DEPLOYMENT"] = "modal"
    from swe_bench_verified import SYSTEM_PROMPT, USER_PROMPT

    def reset(base: str) -> str:
        return " && ".join(["cd /testbed", "git restore .", "git reset --hard",
                            f"git checkout {base}", "git clean -fdq"])

    want = set(args.difficulties) if args.difficulties is not None else None
    # R2E-Gym/SWE-Bench-Verified carries the eval fields but no difficulty; take it from princeton.
    difficulty = {ex["instance_id"]: ex.get("difficulty")
                  for ex in load_shared_dataset("princeton-nlp/SWE-bench_Verified", split="test")}

    rows, total = [], 0
    for ex in load_shared_dataset("R2E-Gym/SWE-Bench-Verified", split="test"):
        total += 1
        iid, d = ex["instance_id"], difficulty.get(ex["instance_id"])
        if want is not None and d not in want:
            continue
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(problem_statement=ex["problem_statement"])},
            ],
            "agent_name": "swe_agent",
            "extra_info": {"tools_kwargs": {
                "env": {"deployment": {"image": f"{MIRROR}/swebench-verified:sweb.eval.x86_64.{iid}"},
                        "post_setup_cmd": reset(ex["base_commit"])},
                "reward": {"name": "swe_bench", "metadata": {**ex, "difficulty": d}},
            }},
        })
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"swebench {'full' if want is None else sorted(want)}: {len(rows)}/{total} -> {args.out}", flush=True)
    return 0


# ── skip-list (regenerate config/bad_instances.json from gate results) ─────────
def _fail_reason(rec: dict) -> str:
    if rec.get("error"):
        return "gate_error"
    b, f = rec.get("buggy_shim") or {}, rec.get("fixed") or {}
    if b.get("n_parsed") == 0 or f.get("n_parsed") == 0:
        return "collection_failed_zero_tests"
    if b.get("resolved") is True:
        return "buggy_did_not_fail_f2p"
    if f.get("resolved") is False:
        return "fixed_did_not_pass_p2p"
    return "f2p_p2p_mismatch"


def cmd_skip_list(args) -> int:
    existing = {e["id"]: e for e in json.loads(CONFIG.read_text()).get("skip", [])} if CONFIG.exists() else {}
    gate_pass, gate_fail = set(), {}
    for path in args.gate:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = rec.get("instance_id")
            if not iid:
                continue
            if rec.get("gate_pass") is True:
                gate_pass.add(iid)
            else:
                entry = {"id": iid, "repo": iid.split("__")[0],
                         "reason": _fail_reason(rec), "source": args.source}
                if args.evidence:
                    entry["evidence"] = args.evidence
                gate_fail[iid] = entry
    coverage = len(gate_pass) + len(gate_fail)
    may_drop = args.drop_passed_report_seed and coverage >= args.expected_total
    merged: dict[str, dict] = {}
    for iid, e in existing.items():
        if e.get("source", "").startswith("report") and may_drop and iid in gate_pass:
            continue
        merged[iid] = e
    merged.update(gate_fail)
    skip = sorted(merged.values(), key=lambda e: (e["repo"], e["id"]))
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    cfg["skip"] = skip
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"gate_pass={len(gate_pass)} gate_fail={len(gate_fail)} coverage={coverage} "
          f"drop_seed={may_drop} total_skip={len(skip)} -> {CONFIG}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Self-contained P2A data builder (subcommands).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("r2e", help="HF -> R2E training parquet (full + .train.parquet)")
    r.add_argument("--out", required=True)
    r.add_argument("--train-out", default=None)
    r.add_argument("--no-skip-filter", action="store_true")
    r.set_defaults(func=cmd_r2e)

    for name, default_diff, help_text in [
        ("swebench-verified", None, "HF -> SWE-bench-Verified full eval set"),
        ("swebench-hard", sorted(HARD_DIFFICULTIES), "HF -> SWE-bench-Verified HARD subset"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--out", required=True)
        sp.add_argument("--difficulties", nargs="*", default=default_diff)
        sp.set_defaults(func=cmd_swebench)

    k = sub.add_parser("skip-list", help="regenerate config/bad_instances.json from gate results")
    k.add_argument("--gate", action="append", default=[], required=True)
    k.add_argument("--drop-passed-report-seed", action="store_true")
    k.add_argument("--expected-total", type=int, default=4503)
    k.add_argument("--source", default="uni_agent_gate")
    k.add_argument("--evidence", default=None)
    k.set_defaults(func=cmd_skip_list)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
