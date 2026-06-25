"""Offline P2A trajectory dashboard artifact builder."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

from p2a.core import BonusMapStore
from p2a.eval_fault_localization import (
    _json_default,
    iter_records,
    score_record,
    summarize,
    summarize_trends,
    write_jsonl,
)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value))


def _badge(label: str, active: bool) -> str:
    state = "on" if active else "off"
    return f'<span class="badge {state}">{html.escape(label)}</span>'


def _graph_topology(item: dict[str, Any]) -> str:
    topology = item.get("graph_topology") or {}
    nodes = topology.get("nodes") or []
    edges = topology.get("edges") or []
    if not nodes:
        return "-"

    node_rows = []
    for node in nodes[:40]:
        line_range = f"{node.get('start_line', '-')}-{node.get('end_line', '-')}"
        node_rows.append(
            "<tr>"
            f"<td>{_fmt(node.get('key'))}</td>"
            f"<td>{_fmt(node.get('file_path'))}:{_fmt(line_range)}</td>"
            f"<td>{_fmt(node.get('normalized_distance'))}</td>"
            f"<td>{_fmt(node.get('node_role'))}</td>"
            f"<td>{_badge('reward', bool(node.get('rewardable', True)))}</td>"
            f"<td>{_fmt(node.get('exclusion_reason'))}</td>"
            f"<td>{_badge('hit', bool(node.get('hit')))}</td>"
            f"<td>{_fmt(node.get('first_step'))}</td>"
            "</tr>"
        )
        if node.get("source_preview"):
            node_rows.append(
                '<tr class="source-row">'
                f'<td colspan="8"><pre>{html.escape(str(node["source_preview"]))}</pre></td>'
                "</tr>"
            )
    edge_rows = "\n".join(
        f"<li>{_fmt(edge[0])} -> {_fmt(edge[1])}</li>"
        for edge in edges[:60]
        if isinstance(edge, list) and len(edge) == 2
    )
    extra_nodes = "" if len(nodes) <= 40 else f'<div class="muted">+{len(nodes) - 40} more nodes in details.jsonl</div>'
    extra_edges = "" if len(edges) <= 60 else f'<div class="muted">+{len(edges) - 60} more edges in details.jsonl</div>'
    return (
        f'<details class="graph"><summary>{len(nodes)} nodes / {len(edges)} edges</summary>'
        '<div class="graph-grid">'
        '<table class="graph-table"><thead><tr><th>Node</th><th>Range</th><th>d</th><th>Role</th><th>Reward</th><th>Reason</th><th>Read</th><th>First step</th></tr></thead>'
        f"<tbody>{''.join(node_rows)}</tbody></table>"
        f"{extra_nodes}"
        f"<ul>{edge_rows}</ul>"
        f"{extra_edges}"
        "</div></details>"
    )


def _node_label(node: dict[str, Any]) -> str:
    labels = []
    if node.get("selected_issue_anchor"):
        labels.append("issue anchor")
    if node.get("root_cause"):
        labels.append("root")
    role = node.get("node_role")
    if role:
        labels.append(str(role))
    return " / ".join(labels) or "-"


def _dependency_graph(item: dict[str, Any]) -> str:
    projection = item.get("chain_projection") or {}
    chain_nodes = projection.get("chain_nodes") or []
    context_nodes = projection.get("context_nodes") or []
    chain_edges = projection.get("chain_edges") or []
    context_edges = projection.get("context_edges") or []
    if not chain_nodes and not context_nodes:
        reason = item.get("not_chain_evaluable_reason") or "no schema-v5 chain projection"
        return f'<div class="muted">{_fmt(reason)}</div>'

    def node_row(node: dict[str, Any]) -> str:
        line_range = f"{node.get('start_line', '-')}-{node.get('end_line', '-')}"
        source = node.get("source_preview")
        source_panel = (
            f'<div class="source-panel"><pre>{html.escape(str(source))}</pre></div>'
            if source
            else '<div class="muted">No source for this node role.</div>'
        )
        group = "context-node" if node.get("group") == "context" else "chain-node"
        return (
            f'<tr class="{group}">'
            "<td>"
            '<details class="node-details">'
            f'<summary>{_fmt(node.get("key"))}</summary>'
            f"{source_panel}"
            "</details>"
            "</td>"
            f"<td>{_fmt(node.get('file_path'))}:{_fmt(line_range)}</td>"
            f"<td>{_fmt(_node_label(node))}</td>"
            f"<td>{_badge('hit', bool(node.get('hit')))}</td>"
            f"<td>{_fmt(node.get('first_step'))}</td>"
            "</tr>"
        )

    def edge_items(edges: list[dict[str, Any]], css_class: str) -> str:
        return "\n".join(
            f'<li class="{css_class}">{_fmt(edge.get("caller"))} -> {_fmt(edge.get("callee"))} '
            f'<span class="muted">{_fmt(edge.get("role_transition"))}</span></li>'
            for edge in edges[:80]
        )

    not_eval = ""
    if not item.get("chain_evaluable"):
        not_eval = f'<div class="muted">Not chain-evaluable: {_fmt(item.get("not_chain_evaluable_reason"))}</div>'
    chain_edge_html = edge_items(chain_edges, "chain-edge") or '<li class="muted">-</li>'
    context_edge_html = edge_items(context_edges, "context-edge") or '<li class="muted">-</li>'
    return (
        f"{not_eval}"
        f'<div class="muted">{len(chain_nodes)} chain nodes, {len(context_nodes)} context nodes, '
        f"{len(chain_edges)} chain edges, {len(context_edges)} context edges</div>"
        '<table class="graph-table dependency-table"><thead><tr>'
        "<th>Node</th><th>Range</th><th>Role</th><th>Read</th><th>First step</th>"
        "</tr></thead><tbody>"
        f"{''.join(node_row(node) for node in context_nodes + chain_nodes)}"
        "</tbody></table>"
        '<div class="edge-grid">'
        f"<section><h4>Chain edges</h4><ul>{chain_edge_html}</ul></section>"
        f"<section><h4>Context edges</h4><ul>{context_edge_html}</ul></section>"
        "</div>"
    )


def _summary_cards(summary: dict[str, Any]) -> str:
    rates = summary.get("rates", {})
    averages = summary.get("averages", {})
    counts = summary.get("counts", {})
    keys = [
        ("Chain coverage", rates.get("chain_graph_coverage")),
        ("Anchor hit", rates.get("anchor_hit_rate")),
        ("Root hit", rates.get("root_hit_rate")),
        ("Chain hit", rates.get("chain_hit_rate")),
        ("Chain recall", rates.get("chain_node_recall")),
        ("Chain precision", rates.get("chain_read_precision")),
        ("Time to anchor", averages.get("time_to_anchor")),
        ("Time to root", averages.get("time_to_root")),
        ("Anchor -> root", averages.get("steps_anchor_to_root")),
        ("Anchor before root", rates.get("anchor_before_root_rate")),
        ("Not evaluable", counts.get("n_not_chain_evaluable")),
    ]
    return "\n".join(
        f'<section class="metric"><div>{html.escape(name)}</div><strong>{_fmt(value)}</strong></section>'
        for name, value in keys
    )


def _trend_rows(summary: dict[str, Any]) -> str:
    trends = summary.get("trends") or []
    if not trends:
        return '<tr><td colspan="10" class="muted">No run-step field found in rollout records.</td></tr>'
    rows = []
    for row in trends[:500]:
        rates = row.get("rates") or {}
        averages = row.get("averages") or {}
        rows.append(
            "<tr>"
            f"<td>{_fmt(row.get('data_source'))}</td>"
            f"<td>{_fmt(row.get('run_step'))}</td>"
            f"<td>{_fmt(row.get('n_records'))}</td>"
            f"<td>{_fmt(rates.get('chain_graph_coverage'))}</td>"
            f"<td>{_fmt(rates.get('anchor_hit_rate'))}</td>"
            f"<td>{_fmt(rates.get('root_hit_rate'))}</td>"
            f"<td>{_fmt(rates.get('chain_hit_rate'))}</td>"
            f"<td>{_fmt(rates.get('chain_node_recall'))}</td>"
            f"<td>{_fmt(averages.get('steps_anchor_to_root'))}</td>"
            f"<td>{_fmt(rates.get('anchor_before_root_rate'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _distribution_panel(summary: dict[str, Any]) -> str:
    distributions = summary.get("distributions") or {}
    not_chain_reasons = distributions.get("not_chain_evaluable_reasons") or {}
    chain_bad_patterns = distributions.get("chain_bad_patterns") or {}
    by_case = summary.get("by_case_type") or {}

    def rows(items: Iterable[tuple[Any, Any]], value_key: str = "value") -> str:
        out = []
        for key, value in items:
            if isinstance(value, dict):
                display = value.get(value_key, value.get("n", value))
            else:
                display = value
            out.append(f"<tr><td>{_fmt(key)}</td><td>{_fmt(display)}</td></tr>")
        return "\n".join(out) or '<tr><td colspan="2" class="muted">-</td></tr>'

    return (
        '<div class="mini-grid">'
        '<section><h3>Not chain-evaluable</h3><table><tbody>'
        f"{rows(not_chain_reasons.items())}"
        "</tbody></table></section>"
        '<section><h3>Chain bad patterns</h3><table><tbody>'
        f"{rows(chain_bad_patterns.items())}"
        "</tbody></table></section>"
        '<section><h3>Case types</h3><table><tbody>'
        f"{rows(by_case.items(), value_key='n')}"
        "</tbody></table></section>"
        "</div>"
    )


def _block_lane(item: dict[str, Any]) -> str:
    blocks = item.get("purpose_blocks") or []
    if not blocks:
        return '<div class="muted">No purpose blocks.</div>'
    parts = []
    for block in blocks[:80]:
        state = "achieved" if block.get("achieved") else "loop" if block.get("loop") else "wasted" if block.get("wasted") else "neutral"
        hit = f" @ step {block['first_hit_step']}" if block.get("first_hit_step") is not None else ""
        distance = f" d={_fmt(block.get('min_distance'))}" if block.get("min_distance") is not None else ""
        parts.append(
            '<div class="block-line">'
            f'<span class="badge {state}">{html.escape(state)}</span>'
            f"<strong>Block {block.get('block_index')}</strong> "
            f"{_fmt(block.get('family'))} {_fmt(block.get('target_path'))} "
            f"steps {_fmt(block.get('step_indices'))}{hit}{distance}"
            "</div>"
        )
    return "\n".join(parts)


def _step_list(item: dict[str, Any]) -> str:
    steps = item.get("step_details") or []
    if not steps:
        return '<div class="muted">No step traces.</div>'
    rows = []
    for step in steps[:120]:
        hits = ", ".join(node["key"] for node in step.get("hit_nodes", [])) or "-"
        rows.append(
            "<tr>"
            f"<td>{_fmt(step.get('step_index'))}</td>"
            f"<td>{_fmt(step.get('family'))}</td>"
            f"<td>{_fmt(step.get('target_path'))}</td>"
            f"<td>{_fmt(step.get('n_reads'))}</td>"
            f"<td>{_fmt(step.get('min_distance'))}</td>"
            f"<td>{_fmt(hits)}</td>"
            "</tr>"
        )
    return (
        '<table class="step-table"><thead><tr><th>Step</th><th>Family</th><th>Target</th>'
        "<th>Reads</th><th>d</th><th>Matched nodes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _trace_details(item: dict[str, Any]) -> str:
    bad = item.get("bad_patterns") or {}
    chain_bad = item.get("chain_bad_patterns") or {}
    bad_labels = " ".join(
        [
            _badge("loop", bool(bad.get("has_loop"))),
            _badge("error spiral", bool(bad.get("error_spiral"))),
        ]
    )
    chain_badges = " ".join(
        _badge(label.replace("_", " "), bool(chain_bad.get(label)))
        for label in (
            "missed_anchor",
            "missed_root_after_anchor",
            "root_before_anchor",
            "chain_stall",
            "chain_read_loop",
            "off_chain_read_spree",
            "error_spiral_on_chain",
        )
    )
    return (
        '<details class="trace"><summary>Open</summary>'
        f'<div class="badges">{_badge("chain evaluable", bool(item.get("chain_evaluable")))} '
        f'{_badge("direct", item.get("chain_case_kind") == "direct")}</div>'
        "<h3>Dependency graph projection</h3>"
        f"{_dependency_graph(item)}"
        "<h3>Chain bad patterns</h3>"
        f'<div class="badges">{chain_badges}</div>'
        "<h3>Trace hygiene patterns</h3>"
        f'<div class="badges">{bad_labels}</div>'
        "<h3>Purpose blocks</h3>"
        f"{_block_lane(item)}"
        "<h3>Step annotations</h3>"
        f"{_step_list(item)}"
        "<h3>Graph topology (full)</h3>"
        f"{_graph_topology(item)}"
        "</details>"
    )


def _record_rows(details: list[dict[str, Any]]) -> str:
    rows = []
    for item in details[:300]:
        badges = " ".join(
            [
                _badge("chain", bool(item.get("chain_hit"))),
                _badge("anchor", bool(item.get("anchor_hit"))),
                _badge("root", bool(item.get("root_hit"))),
                _badge("evaluable", bool(item.get("chain_evaluable"))),
                _badge("direct", item.get("chain_case_kind") == "direct"),
            ]
        )
        rows.append(
            "<tr>"
            f"<td>{_fmt(item.get('record_index'))}</td>"
            f"<td>{_fmt(item.get('instance_id'))}</td>"
            f"<td>{badges}</td>"
            f"<td>{_fmt(item.get('chain_node_recall'))}</td>"
            f"<td>{_fmt(item.get('chain_read_precision'))}</td>"
            f"<td>{_fmt(item.get('first_anchor_step'))}</td>"
            f"<td>{_fmt(item.get('first_root_step'))}</td>"
            f"<td>{_fmt(item.get('steps_anchor_to_root'))}</td>"
            f"<td>{_fmt(item.get('not_chain_evaluable_reason'))}</td>"
            f"<td>{_trace_details(item)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_dashboard(summary: dict[str, Any], details: list[dict[str, Any]]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>P2A trajectory dashboard</title>
<style>
body {{ margin: 0; font: 14px/1.45 system-ui, sans-serif; background: #f7f8fb; color: #172033; }}
header {{ padding: 24px 32px 16px; background: #162033; color: white; }}
h1 {{ margin: 0 0 6px; font-size: 24px; letter-spacing: 0; }}
main {{ padding: 24px 32px 40px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.metric {{ background: white; border: 1px solid #d9dee8; border-radius: 6px; padding: 12px; }}
.metric div {{ color: #667085; font-size: 12px; }}
.metric strong {{ display: block; margin-top: 6px; font-size: 22px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee8; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e6e9f0; text-align: left; vertical-align: top; }}
th {{ color: #475467; font-size: 12px; background: #f0f3f8; }}
.badge {{ display: inline-block; margin: 0 4px 4px 0; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
.badge.on {{ background: #d8f3dc; color: #166534; }}
.badge.off {{ background: #eceff4; color: #667085; }}
.badge.achieved {{ background: #d8f3dc; color: #166534; }}
.badge.wasted {{ background: #fee2e2; color: #991b1b; }}
.badge.loop {{ background: #fef3c7; color: #92400e; }}
.badge.neutral {{ background: #eceff4; color: #667085; }}
.graph summary {{ cursor: pointer; color: #2453a6; font-weight: 600; }}
.graph-grid {{ margin-top: 8px; min-width: 620px; }}
.graph-table {{ font-size: 12px; }}
.source-row pre {{ margin: 0; padding: 8px; overflow-x: auto; background: #111827; color: #e5e7eb; border-radius: 4px; }}
.source-panel pre {{ margin: 8px 0 0; padding: 8px; overflow-x: auto; background: #111827; color: #e5e7eb; border-radius: 4px; }}
.dependency-table .context-node {{ color: #667085; background: #f8fafc; }}
.dependency-table .chain-node {{ background: #fff; }}
.node-details summary {{ cursor: pointer; color: #2453a6; font-weight: 600; }}
.edge-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
.edge-grid h4 {{ margin: 8px 0; font-size: 12px; color: #344054; }}
.chain-edge {{ font-weight: 600; }}
.context-edge {{ color: #667085; border-left: 2px dashed #cbd5e1; padding-left: 6px; }}
.mini-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
.mini-grid section {{ background: white; border: 1px solid #d9dee8; border-radius: 6px; padding: 12px; }}
.mini-grid h3, .trace h3 {{ margin: 8px 0; font-size: 13px; color: #344054; }}
.trace summary {{ cursor: pointer; color: #2453a6; font-weight: 600; }}
.block-line {{ margin: 4px 0; }}
.step-table {{ font-size: 12px; }}
.muted {{ color: #667085; font-size: 12px; margin: 6px 0; }}
.section-title {{ margin: 28px 0 10px; font-size: 18px; }}
</style>
</head>
<body>
<header>
<h1>P2A trajectory dashboard</h1>
<div>{_fmt(summary.get("source"))} against {_fmt(summary.get("bonus_map_dir"))}</div>
</header>
<main>
<div class="grid">
{_summary_cards(summary)}
</div>
<h2 class="section-title">Trend panel</h2>
<table>
<thead>
<tr><th>Data source</th><th>Step</th><th>N</th><th>Chain coverage</th><th>Anchor hit</th><th>Root hit</th><th>Chain hit</th><th>Chain recall</th><th>Anchor -> root</th><th>Anchor before root</th></tr>
</thead>
<tbody>
{_trend_rows(summary)}
</tbody>
</table>
<h2 class="section-title">Aggregate dashboard</h2>
{_distribution_panel(summary)}
<h2 class="section-title">Trace drill-down</h2>
<table>
<thead>
<tr><th>#</th><th>Instance</th><th>Badges</th><th>Chain recall</th><th>Chain precision</th><th>Anchor step</th><th>Root step</th><th>Anchor -> root</th><th>Not-chain reason</th><th>Trace</th></tr>
</thead>
<tbody>
{_record_rows(details)}
</tbody>
</table>
</main>
</body>
</html>
"""


def build_dashboard(
    rollouts: Path,
    bonus_map_dir: Path,
    out_dir: Path,
    *,
    tracking_mode: str = "view_and_bash",
    near_threshold: float = 0.5,
    m_max: float = 3.0,
) -> dict[str, Path]:
    bonus_maps = BonusMapStore(str(bonus_map_dir))
    records = list(iter_records(rollouts))
    details = [
        score_record(
            record,
            index=index,
            bonus_maps=bonus_maps,
            tracking_mode=tracking_mode,
            near_threshold=near_threshold,
            m_max=m_max,
        )
        for index, record in enumerate(records)
    ]
    summary = summarize(
        details,
        source=rollouts,
        bonus_map_dir=bonus_map_dir,
        tracking_mode=tracking_mode,
        near_threshold=near_threshold,
        m_max=m_max,
    )
    summary["trends"] = summarize_trends(
        details,
        tracking_mode=tracking_mode,
        near_threshold=near_threshold,
        m_max=m_max,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    details_path = out_dir / "details.jsonl"
    summary_path = out_dir / "summary.json"
    html_path = out_dir / "index.html"
    write_jsonl(details_path, details)
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    html_path.write_text(render_dashboard(summary, details), encoding="utf-8")
    return {"details": details_path, "summary": summary_path, "html": html_path}
