#!/usr/bin/env python3
"""Generate a compact HTML/Markdown report for a bonus-map directory."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DYNAMIC_TRACEABLE = {"standard", "direct"}
STATIC_TRACEABLE = {"newly_created"}
CASE_ORDER = [
    "standard",
    "direct",
    "newly_created",
    "all_pass",
    "no_trace",
    "no_f2p",
    "no_gt",
    "no_callable",
    "static_fallback",
]


def _load_jsons(root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            items.append({"instance_id": path.stem, "case_type": "parse_error", "error": True, "error_message": str(exc)})
            continue
        data.setdefault("instance_id", path.stem)
        items.append(data)
    return items


def _pct(num: int | float, den: int | float) -> float:
    return 0.0 if not den else float(num) * 100.0 / float(den)


def _median(values: list[int | float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _mean(values: list[int | float]) -> float | None:
    return None if not values else float(statistics.mean(values))


def _p95(values: list[int | float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return float(ordered[idx])


def analyze(root: Path, *, expected_total: int | None = None) -> dict[str, Any]:
    items = _load_jsons(root)
    total = len(items)
    case_counts = Counter(str(item.get("case_type") or "unknown") for item in items)
    reason_counts = Counter(str(item.get("reason_code") or item.get("case_type") or "unknown") for item in items)

    dynamic_traceable = sum(case_counts[c] for c in DYNAMIC_TRACEABLE)
    static_traceable = sum(case_counts[c] for c in STATIC_TRACEABLE)
    errors = sum(1 for item in items if bool(item.get("error")))

    hop_values = [
        int(item["hop_max"])
        for item in items
        if str(item.get("case_type")) in DYNAMIC_TRACEABLE and isinstance(item.get("hop_max"), int)
    ]
    f2p_counts = [
        int(item["f2p_trace_count"])
        for item in items
        if str(item.get("case_type")) in DYNAMIC_TRACEABLE and isinstance(item.get("f2p_trace_count"), int)
    ]

    uni_agent = [item for item in items if item.get("sandbox_backend") == "uni_agent"]
    checkout_attempted = [item for item in items if item.get("buggy_checkout_ref")]
    checkout_ok = [item for item in checkout_attempted if item.get("buggy_checkout_exit") == 0 and item.get("buggy_checkout_head")]
    checkout_ref_oldish = [
        item
        for item in checkout_ok
        if str(item.get("buggy_checkout_ref", "")).endswith("^")
        or str(item.get("buggy_checkout_ref", "")) == str(item.get("old_commit_hash", ""))
    ]

    examples: dict[str, list[str]] = {}
    for case in CASE_ORDER + ["unknown", "parse_error"]:
        examples[case] = [str(item.get("instance_id")) for item in items if str(item.get("case_type") or "unknown") == case][:8]

    return {
        "bonus_maps_dir": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expected_total": expected_total,
        "completed": total,
        "missing": None if expected_total is None else max(expected_total - total, 0),
        "case_counts": dict(sorted(case_counts.items())),
        "reason_counts": dict(reason_counts.most_common()),
        "dynamic_traceable": dynamic_traceable,
        "static_traceable": static_traceable,
        "training_relevant": dynamic_traceable + static_traceable,
        "errors": errors,
        "rates": {
            "dynamic_traceable_pct": _pct(dynamic_traceable, total),
            "static_traceable_pct": _pct(static_traceable, total),
            "training_relevant_pct": _pct(dynamic_traceable + static_traceable, total),
            "error_pct": _pct(errors, total),
            "checkout_ok_pct": _pct(len(checkout_ok), len(checkout_attempted)),
        },
        "hop_stats": {
            "count": len(hop_values),
            "mean": _mean(hop_values),
            "median": _median(hop_values),
            "p95": _p95(hop_values),
            "max": max(hop_values) if hop_values else None,
        },
        "f2p_trace_stats": {
            "count": len(f2p_counts),
            "mean": _mean(f2p_counts),
            "median": _median(f2p_counts),
            "max": max(f2p_counts) if f2p_counts else None,
        },
        "buggy_state": {
            "uni_agent_backend_count": len(uni_agent),
            "checkout_attempted": len(checkout_attempted),
            "checkout_ok": len(checkout_ok),
            "checkout_ref_oldish": len(checkout_ref_oldish),
            "sample": [
                {
                    "instance_id": item.get("instance_id"),
                    "buggy_checkout_ref": item.get("buggy_checkout_ref"),
                    "buggy_checkout_head": item.get("buggy_checkout_head"),
                    "case_type": item.get("case_type"),
                }
                for item in checkout_ok[:12]
            ],
        },
        "examples": examples,
    }


def _fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _case_rows(analysis: dict[str, Any]) -> str:
    total = int(analysis["completed"])
    counts = Counter(analysis["case_counts"])
    ordered = [case for case in CASE_ORDER if counts.get(case)] + sorted(set(counts) - set(CASE_ORDER))
    rows = []
    for case in ordered:
        count = int(counts[case])
        rows.append(
            f"<tr><td><code>{html.escape(case)}</code></td><td class='num'>{count}</td>"
            f"<td class='num'>{_pct(count, total):.1f}%</td></tr>"
        )
    return "\n".join(rows)


def _reason_rows(analysis: dict[str, Any]) -> str:
    total = int(analysis["completed"])
    rows = []
    for reason, count in list(analysis["reason_counts"].items())[:30]:
        rows.append(
            f"<tr><td><code>{html.escape(reason)}</code></td><td class='num'>{count}</td>"
            f"<td class='num'>{_pct(int(count), total):.1f}%</td></tr>"
        )
    return "\n".join(rows)


def render_html(analysis: dict[str, Any]) -> str:
    title = "P2A Bonus Map Full-Corpus Analysis"
    completed = int(analysis["completed"])
    expected = analysis.get("expected_total")
    completed_text = f"{completed}" if expected is None else f"{completed} / {expected}"
    rates = analysis["rates"]
    hop = analysis["hop_stats"]
    buggy = analysis["buggy_state"]
    case_labels = list(analysis["case_counts"].keys())
    case_values = [analysis["case_counts"][k] for k in case_labels]

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;--blue:#58a6ff;--green:#3fb950;--red:#f85149;--orange:#d29922}}
*{{box-sizing:border-box}} body{{margin:0 auto;max-width:1180px;padding:24px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.55}}
h1{{font-size:1.9rem;margin:0 0 6px}} h2{{font-size:1.25rem;margin:28px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border)}} code{{background:#21262d;padding:1px 5px;border-radius:4px}}
.subtitle{{color:var(--muted);margin-bottom:22px}} .grid{{display:grid;gap:14px}} .grid4{{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}} .grid2{{grid-template-columns:repeat(auto-fit,minmax(360px,1fr))}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:18px}} .label{{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.4px}} .value{{font-size:1.9rem;font-weight:700;margin-top:3px}} .green{{color:var(--green)}} .red{{color:var(--red)}} .orange{{color:var(--orange)}} .muted{{color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:.9rem}} th,td{{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left}} th{{color:var(--muted);font-size:.78rem;text-transform:uppercase}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.chart{{height:300px}} .note{{border-left:3px solid var(--blue);background:rgba(88,166,255,.08);padding:10px 12px;border-radius:0 6px 6px 0;color:#c9d1d9}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<p class="subtitle">Data source: <code>{html.escape(analysis["bonus_maps_dir"])}</code> &middot; Generated {html.escape(analysis["generated_at"])}</p>
<div class="grid grid4">
  <div class="card"><div class="label">Completed</div><div class="value">{completed_text}</div><div class="muted">{analysis.get("missing", "n/a")} missing</div></div>
  <div class="card"><div class="label">Dynamic Traceable</div><div class="value green">{analysis["dynamic_traceable"]}</div><div class="muted">{rates["dynamic_traceable_pct"]:.1f}% standard/direct</div></div>
  <div class="card"><div class="label">Training-Relevant</div><div class="value green">{analysis["training_relevant"]}</div><div class="muted">{rates["training_relevant_pct"]:.1f}% incl newly_created</div></div>
  <div class="card"><div class="label">Errors</div><div class="value red">{analysis["errors"]}</div><div class="muted">{rates["error_pct"]:.1f}%</div></div>
</div>
<h2>Buggy-State Validation</h2>
<div class="card">
  <p class="note">This report is valid for P2A only when Uni-Agent/veFaaS sandboxes restore the buggy old commit before tracing. Checkout success below is computed from bonus-map diagnostics.</p>
  <table><tbody>
    <tr><th>Uni-Agent backend outputs</th><td class="num">{buggy["uni_agent_backend_count"]}</td></tr>
    <tr><th>Buggy checkout attempted</th><td class="num">{buggy["checkout_attempted"]}</td></tr>
    <tr><th>Buggy checkout succeeded</th><td class="num">{buggy["checkout_ok"]} ({rates["checkout_ok_pct"]:.1f}%)</td></tr>
    <tr><th>Old-commit refs observed</th><td class="num">{buggy["checkout_ref_oldish"]}</td></tr>
  </tbody></table>
</div>
<h2>Case Distribution</h2>
<div class="grid grid2">
  <div class="card"><canvas id="caseChart" class="chart"></canvas></div>
  <div class="card"><table><thead><tr><th>Case</th><th class="num">Count</th><th class="num">%</th></tr></thead><tbody>{_case_rows(analysis)}</tbody></table></div>
</div>
<h2>Reason Codes</h2>
<div class="card"><table><thead><tr><th>Reason</th><th class="num">Count</th><th class="num">%</th></tr></thead><tbody>{_reason_rows(analysis)}</tbody></table></div>
<h2>Traceable Stats</h2>
<div class="grid grid4">
  <div class="card"><div class="label">Hop Mean</div><div class="value">{_fmt_num(hop["mean"])}</div></div>
  <div class="card"><div class="label">Hop Median</div><div class="value">{_fmt_num(hop["median"])}</div></div>
  <div class="card"><div class="label">Hop P95</div><div class="value">{_fmt_num(hop["p95"])}</div></div>
  <div class="card"><div class="label">Hop Max</div><div class="value">{_fmt_num(hop["max"], 0)}</div></div>
</div>
<script>
new Chart(document.getElementById('caseChart'), {{
  type: 'doughnut',
  data: {{labels: {json.dumps(case_labels)}, datasets: [{{data: {json.dumps(case_values)}, backgroundColor: ['#3fb950','#58a6ff','#bc8cff','#d29922','#f85149','#f0883e','#8b949e','#6e7681']}}]}},
  options: {{plugins: {{legend: {{position: 'bottom', labels: {{color: '#e6edf3'}}}}}}}}
}});
</script>
</body>
</html>
"""


def render_markdown(analysis: dict[str, Any]) -> str:
    total = int(analysis["completed"])
    expected = analysis.get("expected_total")
    completed_text = f"{total}" if expected is None else f"{total} / {expected}"
    lines = [
        "# P2A Bonus Map Full-Corpus Analysis",
        "",
        f"**Data**: `{analysis['bonus_maps_dir']}`",
        f"**Generated**: {analysis['generated_at']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Completed | {completed_text} |",
        f"| Dynamic traceable | {analysis['dynamic_traceable']} ({analysis['rates']['dynamic_traceable_pct']:.1f}%) |",
        f"| Training-relevant | {analysis['training_relevant']} ({analysis['rates']['training_relevant_pct']:.1f}%) |",
        f"| Errors | {analysis['errors']} ({analysis['rates']['error_pct']:.1f}%) |",
        f"| Buggy checkout OK | {analysis['buggy_state']['checkout_ok']} / {analysis['buggy_state']['checkout_attempted']} |",
        "",
        "## Case Distribution",
        "",
        "| Case | Count | % |",
        "|---|---:|---:|",
    ]
    for case, count in sorted(analysis["case_counts"].items()):
        lines.append(f"| `{case}` | {count} | {_pct(int(count), total):.1f}% |")
    lines.extend(["", "## Reason Codes", "", "| Reason | Count | % |", "|---|---:|---:|"])
    for reason, count in analysis["reason_counts"].items():
        lines.append(f"| `{reason}` | {count} | {_pct(int(count), total):.1f}% |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bonus_maps_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("report/bonus-map-full-analysis"))
    parser.add_argument("--expected-total", type=int, default=None)
    args = parser.parse_args()

    analysis = analyze(args.bonus_maps_dir, expected_total=args.expected_total)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "index.html").write_text(render_html(analysis), encoding="utf-8")
    (args.output_dir / "report.md").write_text(render_markdown(analysis), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'index.html'}")
    print(f"Wrote {args.output_dir / 'report.md'}")
    print(f"Wrote {args.output_dir / 'analysis.json'}")


if __name__ == "__main__":
    main()
