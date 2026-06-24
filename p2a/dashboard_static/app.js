const state = {
  snapshot: window.__P2A_DASHBOARD_SNAPSHOT__ || null,
  activeTab: "overview",
  selectedExperimentKey: null,
  selectedTraceKey: null,
  selectedStepIndex: 0,
  traceQuery: "",
  refreshTimer: null,
};

function esc(value) {
  return String(value ?? "-")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  return String(value);
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function token(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (value < 1000) return String(Math.round(value));
  if (value < 1000000) return `${(value / 1000).toFixed(1)}k`;
  return `${(value / 1000000).toFixed(1)}m`;
}

function money(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `$${Number(value).toFixed(4)}`;
}

function badge(label, active = true, tone = "") {
  return `<span class="badge ${active ? tone : ""}">${esc(label)}</span>`;
}

function table(headers, rows) {
  if (!rows.length) return '<div class="empty">No rows.</div>';
  return `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

function rowKey(detail) {
  return `${detail.experiment_key || "experiment"}::${detail.instance_id || detail.record_index}`;
}

function experimentRows(snapshot) {
  return snapshot?.experiments || [];
}

function selectedExperiment(snapshot) {
  return experimentRows(snapshot).find((row) => row.experiment_key === state.selectedExperimentKey) || null;
}

function ensureSelection(snapshot) {
  const experiments = experimentRows(snapshot);
  if (!experiments.length) {
    state.selectedExperimentKey = null;
    state.selectedTraceKey = null;
    return;
  }
  if (!state.selectedExperimentKey || !experiments.some((row) => row.experiment_key === state.selectedExperimentKey)) {
    state.selectedExperimentKey = experiments.length === 1 ? experiments[0].experiment_key : null;
  }
  if (!state.selectedExperimentKey) {
    state.selectedTraceKey = null;
    state.selectedStepIndex = 0;
    return;
  }
  const details = filteredDetails(snapshot);
  if (!details.length) {
    state.selectedTraceKey = null;
    state.selectedStepIndex = 0;
    return;
  }
  if (!state.selectedTraceKey || !details.some((detail) => rowKey(detail) === state.selectedTraceKey)) {
    state.selectedTraceKey = rowKey(details[0]);
    state.selectedStepIndex = 0;
  }
}

async function loadSnapshot() {
  if (window.__P2A_DASHBOARD_SNAPSHOT__) {
    state.snapshot = window.__P2A_DASHBOARD_SNAPSHOT__;
    window.__P2A_DASHBOARD_SNAPSHOT__ = null;
    render();
    return;
  }
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.snapshot = await response.json();
  } catch (_error) {
    if (!state.snapshot) {
      try {
        const response = await fetch("snapshot.json", { cache: "no-store" });
        if (response.ok) state.snapshot = await response.json();
      } catch (_fallback) {
        state.snapshot = null;
      }
    }
  }
  render();
}

function renderSources(snapshot) {
  const sources = snapshot?.sources || [];
  const text = sources.map((item) => `${item.kind}: ${item.path}`).join("  |  ");
  document.getElementById("source-line").textContent = text || "No source loaded";
}

function metricCard(label, value, formatter = fmt) {
  return `<section class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(formatter(value))}</div></section>`;
}

function renderSelectedExperiment(snapshot) {
  const selected = selectedExperiment(snapshot);
  const label = selected
    ? `${selected.source_kind || "-"} / ${selected.experiment_id || "-"} / ${selected.model_label || "-"} / ${selected.dataset || "-"}`
    : "None";
  document.getElementById("selected-experiment").textContent = label;
}

function renderSummary(snapshot) {
  const summary = snapshot?.summary || {};
  const rates = summary.rates || {};
  const avg = summary.averages || {};
  const counts = summary.counts || {};
  const cards = [
    ["Experiments", (snapshot.experiments || []).length, fmt],
    ["Records", counts.n_records, fmt],
    ["Models", (snapshot.model_metrics || []).length, fmt],
    ["Runs", (snapshot.runs || []).length, fmt],
    ["Anchor hit", rates.anchor_hit_rate, pct],
    ["Root hit", rates.root_hit_rate, pct],
    ["Node recall", rates.chain_node_recall ?? rates.avg_node_recall, pct],
    ["Read precision", rates.chain_read_precision ?? rates.avg_read_precision, pct],
    ["Reverse order", rates.reverse_order_rate, pct],
    ["Miracle", rates.miracle_rate_over_gt_hits, pct],
    ["Block achieve", rates.block_achieve_rate, pct],
    ["Not evaluable", counts.n_not_chain_evaluable, fmt],
    ["Time to anchor", avg.time_to_anchor, fmt],
    ["Time to root", avg.time_to_root, fmt],
  ];
  document.getElementById("summary-grid").innerHTML = cards.map(([label, value, formatter]) => metricCard(label, value, formatter)).join("");
}

function progress(row) {
  const done = row.done ?? row.detail_count ?? 0;
  const target = row.target ?? row.detail_count ?? 0;
  return `${done}/${target}`;
}

function renderExperiments(snapshot) {
  const rows = experimentRows(snapshot).map((row) => {
    const selected = row.experiment_key === state.selectedExperimentKey;
    return `<tr class="clickable ${selected ? "is-selected" : ""}" data-experiment-key="${esc(row.experiment_key)}">
      <td><button class="select-exp" type="button" data-experiment-key="${esc(row.experiment_key)}">${selected ? "Selected" : "Inspect"}</button></td>
      <td>${esc(row.source_kind)}</td>
      <td>${esc(row.experiment_id)}</td>
      <td>${esc(row.provider_source)}</td>
      <td>${esc(row.dataset)}</td>
      <td>${esc(row.model_label)}</td>
      <td>${esc(progress(row))}</td>
      <td>${esc(pct(row.resolved_rate))}</td>
      <td>${esc(pct(row.root_hit_rate))}</td>
      <td>${esc(pct(row.chain_node_recall))}</td>
      <td>${esc(pct(row.read_precision))}</td>
      <td>${esc(row.trajectory_count ?? 0)}</td>
    </tr>`;
  });
  document.getElementById("experiment-table").innerHTML = table(
    ["", "Kind", "Experiment", "Provider", "Dataset", "Model", "Done", "Resolved", "Root", "Recall", "Precision", "Traj"],
    rows
  );
  document.querySelectorAll(".select-exp, #experiment-table tr.clickable").forEach((el) => {
    el.addEventListener("click", () => {
      const key = el.dataset.experimentKey;
      if (!key) return;
      state.selectedExperimentKey = key;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      setTab(el.classList.contains("select-exp") ? "traces" : state.activeTab);
      render();
    });
  });
}

function scopedRows(rows) {
  if (!state.selectedExperimentKey) return [];
  return (rows || []).filter((row) => row.experiment_key === state.selectedExperimentKey);
}

function renderTrend(snapshot) {
  const selected = selectedExperiment(snapshot);
  const trends = snapshot?.summary?.trends || [];
  const rows = trends
    .filter((row) => !selected || row.data_source === selected.dataset)
    .map((row) => {
      const rates = row.rates || {};
      const avg = row.averages || {};
      return `<tr>
        <td>${esc(row.data_source)}</td>
        <td>${esc(row.run_step)}</td>
        <td>${esc(row.n_records)}</td>
        <td>${esc(pct(rates.chain_graph_coverage))}</td>
        <td>${esc(pct(rates.anchor_hit_rate))}</td>
        <td>${esc(pct(rates.root_hit_rate))}</td>
        <td>${esc(pct(rates.chain_node_recall))}</td>
        <td>${esc(pct(rates.chain_read_precision))}</td>
        <td>${esc(fmt(avg.steps_anchor_to_root))}</td>
        <td>${esc(pct(rates.anchor_before_root_rate))}</td>
      </tr>`;
    });
  document.getElementById("trend-table").innerHTML = table(
    ["Data source", "Step", "N", "Coverage", "Anchor", "Root", "Recall", "Precision", "Anchor-to-root", "Ordered"],
    rows
  );
}

function miniTable(title, mapping) {
  const rows = Object.entries(mapping || {}).map(([key, value]) => {
    const shown = value && typeof value === "object" ? (value.n ?? value.value ?? JSON.stringify(value)) : value;
    return `<tr><td>${esc(key)}</td><td>${esc(fmt(shown))}</td></tr>`;
  });
  return `<section class="mini-panel"><h3>${esc(title)}</h3>${table(["Key", "Value"], rows)}</section>`;
}

function renderDistributions(snapshot) {
  const dist = snapshot?.summary?.distributions || {};
  const byCase = snapshot?.summary?.by_case_type || {};
  document.getElementById("distribution-grid").innerHTML = [
    miniTable("Not chain-evaluable", dist.not_chain_evaluable_reasons),
    miniTable("Chain bad patterns", dist.chain_bad_patterns),
    miniTable("Case types", byCase),
  ].join("");
}

function renderModels(snapshot) {
  if (!state.selectedExperimentKey) {
    document.getElementById("model-table").innerHTML = '<div class="empty">Select an experiment in Overview before comparing model KPIs.</div>';
    return;
  }
  const rows = scopedRows(snapshot?.model_metrics || []);
  const hasCacheWrite = rows.some((row) => row.cache_write_rate !== null && row.cache_write_rate !== undefined);
  const outcomeRows = rows.map((row) => `<tr>
    <td>${esc(row.model_label)}</td><td>${esc(progress(row))}</td>
    <td>${esc(pct(row.resolved_rate))}</td><td>${esc(pct(row.reward_rate))}</td>
    <td>${esc(row.errors || 0)}</td><td>${esc(row.pending || 0)}</td>
  </tr>`);
  const linkerRows = rows.map((row) => `<tr>
    <td>${esc(row.model_label)}</td><td>${esc(pct(row.p2a_read_rate))}</td>
    <td>${esc(pct(row.avg_read_precision))}</td><td>${esc(pct(row.avg_node_recall))}</td><td>${esc(pct(row.avg_hit_f1))}</td>
    <td>${esc(pct(row.anchor_hit_rate))}</td><td>${esc(pct(row.root_hit_rate))}</td><td>${esc(pct(row.avg_chain_node_recall))}</td>
    <td>${esc(pct(row.avg_chain_read_precision))}</td><td>${esc(fmt(row.avg_first_anchor_step, 1))}</td><td>${esc(fmt(row.avg_first_root_step, 1))}</td>
  </tr>`);
  const orderRows = rows.map((row) => `<tr>
    <td>${esc(row.model_label)}</td><td>${esc(fmt(row.avg_order_score))}</td><td>${esc(pct(row.reverse_order_rate))}</td>
    <td>${esc(pct(row.miracle_rate))}</td><td>${esc(fmt(row.avg_miracle_severity))}</td>
    <td>${esc(fmt(row.avg_block_order_score))}</td><td>${esc(pct(row.block_reverse_order_rate))}</td><td>${esc(pct(row.block_miracle_rate))}</td>
  </tr>`);
  const blockRows = rows.map((row) => `<tr>
    <td>${esc(row.model_label)}</td><td>${esc(fmt(row.avg_blocks_per_trace, 1))}</td><td>${esc(pct(row.block_achieve_rate))}</td>
    <td>${esc(pct(row.block_waste_rate))}</td><td>${esc(pct(row.block_loop_rate))}</td><td>${esc(pct(row.loop_trace_rate))}</td>
    <td>${esc(pct(row.error_spiral_rate))}</td><td>${esc(Object.keys(row.chain_bad_patterns || {}).join(", ") || "-")}</td>
    <td>${esc(Object.keys(row.not_chain_evaluable_reasons || {}).join(", ") || "-")}</td>
  </tr>`);
  const efficiencyHeaders = ["Model", "Turns", "Tools", "Wall", "In", "Out", "Reason", "Cost", "Cache hit"];
  if (hasCacheWrite) efficiencyHeaders.push("Cache write");
  const efficiencyRows = rows.map((row) => {
    const cells = [
      row.model_label,
      fmt(row.avg_turns, 1),
      fmt(row.avg_tool_calls, 1),
      fmt(row.avg_wall_time, 1),
      token(row.avg_input_tokens),
      token(row.avg_output_tokens),
      token(row.avg_reasoning_tokens),
      money(row.total_cost),
      pct(row.cache_hit_rate),
    ];
    if (hasCacheWrite) cells.push(pct(row.cache_write_rate));
    return `<tr>${cells.map((cell) => `<td>${esc(cell)}</td>`).join("")}</tr>`;
  });
  document.getElementById("model-table").innerHTML = `
    <div class="kpi-stack">
      <section><h3>Task outcome</h3>${table(["Model", "Done", "Resolved", "Reward", "Errors", "Pending"], outcomeRows)}</section>
      <section><h3>Linker quality</h3>${table(["Model", "Recovered read", "Read precision", "Node recall", "F1", "Anchor", "Root", "Chain recall", "Chain precision", "First anchor", "First root"], linkerRows)}</section>
      <section><h3>Topology and order</h3>${table(["Model", "Order", "Reverse order", "Miracle", "Miracle severity", "Block order", "Block reverse", "Block miracle"], orderRows)}</section>
      <section><h3>Purpose blocks and bad patterns</h3>${table(["Model", "Blocks", "Achieved", "Wasted", "Loop blocks", "Loop trace", "Error spiral", "Chain flags", "Not evaluable"], blockRows)}</section>
      <section><h3>Efficiency, cost, cache</h3>${table(efficiencyHeaders, efficiencyRows)}</section>
    </div>`;
}

function renderRuns(snapshot) {
  const selected = selectedExperiment(snapshot);
  if (!selected) {
    document.getElementById("run-list").innerHTML = '<div class="empty">Select an experiment in Overview before inspecting run logs.</div>';
    return;
  }
  const runs = (snapshot?.runs || []).filter((run) => {
    const path = String(run.path || "");
    return !selected.experiment_id || path.includes(selected.experiment_id) || path.includes(selected.model_label || "");
  });
  const cards = runs.map((run) => {
    const statusTone = run.status === "completed" ? "ok" : run.status === "running" || run.status === "verify" ? "warn" : "";
    const files = (run.files || []).slice(0, 8).map((name) => `<span class="badge">${esc(name)}</span>`).join("");
    const log = run.log_excerpt ? `<pre class="log">${esc(run.log_excerpt.slice(-6000))}</pre>` : '<div class="muted">No run.log tail.</div>';
    return `<article class="run-card">
      <div class="run-head"><div><div class="run-title">${esc(run.run_id)}</div><div class="run-meta">${esc(run.path)}</div></div>${badge(run.status || "unknown", true, statusTone)}</div>
      <div class="run-meta">Updated ${run.last_update ? new Date(run.last_update * 1000).toLocaleString() : "-"}</div>
      <div>${files}</div>${log}
    </article>`;
  });
  document.getElementById("run-list").innerHTML = cards.join("") || '<div class="empty">No run directories for the selected experiment.</div>';
}

function traceBlob(detail) {
  return JSON.stringify({
    instance_id: detail.instance_id,
    files: detail.read_files,
    bad: detail.bad_patterns,
    chain_bad: detail.chain_bad_patterns,
    reason: detail.not_chain_evaluable_reason,
  }).toLowerCase();
}

function filteredDetails(snapshot) {
  const query = state.traceQuery.trim().toLowerCase();
  return (snapshot?.details || [])
    .filter((detail) => !state.selectedExperimentKey || detail.experiment_key === state.selectedExperimentKey)
    .filter((detail) => !query || traceBlob(detail).includes(query));
}

function selectedDetail(snapshot) {
  const details = filteredDetails(snapshot);
  return details.find((detail) => rowKey(detail) === state.selectedTraceKey) || details[0] || null;
}

function roleTone(node) {
  const role = node?.node_role || "";
  if (role === "root_cause") return "root";
  if (role === "symptom") return "symptom";
  if (role === "intermediate") return "chain";
  return "context";
}

function stepTone(step, detail) {
  const scored = step?.scored || step || {};
  const nodes = scored.hit_nodes || [];
  if (nodes.some((node) => node.node_role === "root_cause")) return "root";
  if (nodes.some((node) => node.node_role === "symptom")) return "symptom";
  if (nodes.length) return "chain";
  if ((scored.n_reads || 0) > 0) return "offmap";
  if ((detail?.bad_patterns || {}).error_spiral) return "bad";
  return "neutral";
}

function graphNodes(detail) {
  const projection = detail?.chain_projection || {};
  const projected = [...(projection.context_nodes || []), ...(projection.chain_nodes || [])];
  if (projected.length) return projected;
  return (detail?.graph_topology?.nodes || []).map((node) => ({ ...node, group: node.rewardable ? "chain" : "context" }));
}

function graphEdges(detail) {
  const projection = detail?.chain_projection || {};
  const edges = [...(projection.context_edges || []), ...(projection.chain_edges || [])];
  if (edges.length) return edges.map((edge) => [edge.caller, edge.callee]);
  return detail?.graph_topology?.edges || [];
}

function renderGraph(detail) {
  const nodes = graphNodes(detail);
  if (!nodes.length) return '<div class="empty">No bonus-map graph available for this instance.</div>';
  const edges = graphEdges(detail);
  const width = 760;
  const height = Math.max(260, Math.min(680, 110 + nodes.length * 34));
  const layers = new Map();
  nodes.forEach((node) => {
    const raw = node.normalized_distance;
    const distance = typeof raw === "number" ? raw : 1;
    const layer = Math.max(0, Math.min(5, Math.round(distance * 5)));
    if (!layers.has(layer)) layers.set(layer, []);
    layers.get(layer).push(node);
  });
  const positions = new Map();
  [...layers.entries()].forEach(([layer, layerNodes]) => {
    const x = 80 + (5 - layer) * 120;
    const gap = height / (layerNodes.length + 1);
    layerNodes.forEach((node, index) => positions.set(node.key, { x, y: gap * (index + 1) }));
  });
  const edgeSvg = edges.map(([caller, callee]) => {
    const a = positions.get(caller);
    const b = positions.get(callee);
    if (!a || !b) return "";
    return `<line class="graph-edge" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"><title>${esc(caller)} -> ${esc(callee)}</title></line>`;
  }).join("");
  const nodeSvg = nodes.map((node) => {
    const pos = positions.get(node.key);
    const tone = roleTone(node);
    const hit = node.hit || node.first_step !== null && node.first_step !== undefined;
    const label = String(node.key || "").split("::").slice(-1)[0].slice(0, 28);
    return `<g class="graph-node ${tone} ${hit ? "hit" : "miss"}" transform="translate(${pos.x},${pos.y})">
      <circle r="18"></circle>
      ${hit ? `<text class="graph-step" y="4">${esc(node.first_step)}</text>` : ""}
      <text class="graph-label" x="26" y="-2">${esc(label)}</text>
      <text class="graph-sub" x="26" y="13">${esc(node.node_role || node.group || "-")}</text>
      <title>${esc(node.key)} ${node.first_step !== null && node.first_step !== undefined ? `first step ${node.first_step}` : "not visited"}</title>
    </g>`;
  }).join("");
  return `<div class="graph-wrap"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Bonus map graph">${edgeSvg}${nodeSvg}</svg></div>`;
}

function traceBadges(detail) {
  const chainBad = detail.chain_bad_patterns || {};
  const bad = detail.bad_patterns || {};
  const items = [
    ["anchor", detail.anchor_hit, "ok"],
    ["root", detail.root_hit, "ok"],
    ["chain", detail.chain_hit, "ok"],
    ["miracle", detail.miracle_step === true || detail.block_miracle_step === true, "warn"],
    ["reverse", (detail.order_score ?? 0) < 0 || (detail.block_order_score ?? 0) < 0, "warn"],
    ["loop", bad.has_loop || chainBad.chain_read_loop, "bad"],
    ["error spiral", bad.error_spiral || chainBad.error_spiral_on_chain, "bad"],
    ["missed anchor", chainBad.missed_anchor, "bad"],
    ["missed root", chainBad.missed_root_after_anchor, "bad"],
  ];
  return items.filter((item) => item[1]).map((item) => badge(item[0], true, item[2])).join("");
}

function renderTraceList(snapshot) {
  const details = filteredDetails(snapshot);
  const rows = details.map((detail) => {
    const selected = rowKey(detail) === state.selectedTraceKey;
    return `<button class="trace-row ${selected ? "is-selected" : ""}" type="button" data-trace-key="${esc(rowKey(detail))}">
      <span class="trace-id">${esc(detail.instance_id || `record-${detail.record_index}`)}</span>
      <span class="trace-meta">root ${pct(detail.root_hit ? 1 : 0)} · recall ${pct(detail.chain_node_recall ?? detail.hit_recall)}</span>
      <span>${traceBadges(detail)}</span>
    </button>`;
  });
  return rows.join("") || '<div class="empty">No trajectories in the selected experiment.</div>';
}

function blockSteps(block) {
  return block.trace_indices || block.step_indices || [];
}

function stepByTraceIndex(detail) {
  const byIndex = new Map();
  (detail.step_inspection || []).forEach((step) => byIndex.set(Number(step.trace_index), step));
  (detail.step_details || []).forEach((scored, index) => {
    const traceIndex = Number(scored.trace_index ?? index);
    if (!byIndex.has(traceIndex)) byIndex.set(traceIndex, { trace_index: traceIndex, step_index: scored.step_index, scored });
  });
  return byIndex;
}

function renderTimeline(detail) {
  const byIndex = stepByTraceIndex(detail);
  const blocks = detail.purpose_blocks || [];
  if (!blocks.length && !byIndex.size) return '<div class="empty">No step-level trace was captured.</div>';
  if (!blocks.length) {
    return `<div class="timeline-grid">${[...byIndex.values()].map((step) => renderStepThumb(step, detail)).join("")}</div>`;
  }
  return blocks.map((block) => {
    const tone = block.achieved ? "ok" : block.loop || block.wasted ? "bad" : "";
    const steps = blockSteps(block).map((idx) => byIndex.get(Number(idx))).filter(Boolean);
    return `<section class="purpose-block ${tone}">
      <div class="block-head">
        <strong>Block ${esc(block.block_index)}</strong>
        ${badge(block.achieved ? "achieved" : block.loop ? "loop" : block.wasted ? "wasted" : "neutral", true, tone)}
        <span>${esc(block.family || "-")} ${esc(block.target_path || "")}</span>
      </div>
      <div class="timeline-grid">${steps.map((step) => renderStepThumb(step, detail)).join("") || '<span class="muted">No captured step in this block.</span>'}</div>
    </section>`;
  }).join("");
}

function renderStepThumb(step, detail) {
  const traceIndex = Number(step.trace_index ?? step.step_index ?? 0);
  const selected = traceIndex === state.selectedStepIndex;
  const tone = stepTone(step, detail);
  const tool = (step.tool_names || [step.scored?.family || "step"]).join("+");
  const target = step.scored?.target_path || "";
  return `<button type="button" class="step-thumb ${tone} ${selected ? "is-selected" : ""}" data-step-index="${traceIndex}">
    <span class="step-num">${esc(step.step_index ?? traceIndex)}</span>
    <span class="step-tool">${esc(tool)}</span>
    <span class="step-target">${esc(target.split("/").slice(-2).join("/") || "no target")}</span>
  </button>`;
}

function jsonBlock(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value ?? "", null, 2);
  return `<pre class="detail-pre">${esc(text || "-")}</pre>`;
}

function renderStepDetail(detail) {
  const byIndex = stepByTraceIndex(detail);
  const step = byIndex.get(Number(state.selectedStepIndex)) || [...byIndex.values()][0];
  if (!step) {
    return `<section class="step-detail"><h3>Trajectory detail</h3><div class="empty">Raw step content was not captured for this artifact.</div></section>`;
  }
  const scored = step.scored || {};
  const hitNodes = (scored.hit_nodes || []).map((node) => `${node.node_role || "node"}: ${node.key}`).join("\n");
  const reads = scored.reads || [];
  return `<section class="step-detail">
    <h3>Step ${esc(step.step_index ?? step.trace_index)}</h3>
    <div class="detail-badges">
      ${badge((step.tool_names || ["step"]).join("+"))}
      ${step.parse_error ? badge("parse error", true, "bad") : ""}
      ${step.exit_reason ? badge(step.exit_reason) : ""}
      ${scored.n_reads ? badge(`${scored.n_reads} recovered reads`, true, "ok") : badge("no recovered read", true, "warn")}
    </div>
    <div class="detail-grid">
      <section><h4>Think / assistant text</h4>${jsonBlock(step.thought || step.response_text || "(empty)")}</section>
      <section><h4>Tool calls</h4>${jsonBlock(step.tool_calls || [])}</section>
      <section><h4>Tool returns</h4>${jsonBlock(step.tool_results || [])}</section>
      <section><h4>Recovered reads</h4>${jsonBlock(reads)}</section>
      <section><h4>Matched bonus-map nodes</h4>${jsonBlock(hitNodes || "No matched bonus-map node.")}</section>
    </div>
  </section>`;
}

function renderTraceInspector(snapshot) {
  const detail = selectedDetail(snapshot);
  if (!state.selectedExperimentKey) {
    document.getElementById("trace-inspector").innerHTML = '<div class="empty">Select an experiment in Overview before inspecting trajectories.</div>';
    return;
  }
  if (!detail) {
    document.getElementById("trace-inspector").innerHTML = '<div class="empty">No trajectory details are available for this experiment.</div>';
    return;
  }
  state.selectedTraceKey = rowKey(detail);
  const selectedSteps = stepByTraceIndex(detail);
  if (!selectedSteps.has(Number(state.selectedStepIndex))) {
    state.selectedStepIndex = Number([...selectedSteps.keys()][0] ?? 0);
  }
  document.getElementById("trace-inspector").innerHTML = `
    <aside class="trace-left">${renderTraceList(snapshot)}</aside>
    <section class="trace-middle">
      <div class="trace-title-line">
        <div><strong>${esc(detail.instance_id || `record-${detail.record_index}`)}</strong><div class="run-meta">${esc(detail.model_label || "-")} · ${esc(detail.run_id || "-")}</div></div>
        <div>${traceBadges(detail)}</div>
      </div>
      ${renderGraph(detail)}
      <h3>Purpose blocks and step timeline</h3>
      ${renderTimeline(detail)}
    </section>
    <aside class="trace-right">${renderStepDetail(detail)}</aside>`;
  document.querySelectorAll(".trace-row").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedTraceKey = button.dataset.traceKey;
      state.selectedStepIndex = 0;
      renderTraceInspector(state.snapshot);
    });
  });
  document.querySelectorAll(".step-thumb").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedStepIndex = Number(button.dataset.stepIndex || 0);
      renderTraceInspector(state.snapshot);
    });
  });
}

function render() {
  const snapshot = state.snapshot;
  if (!snapshot) {
    document.getElementById("summary-grid").innerHTML = '<div class="empty">Dashboard data is not available.</div>';
    return;
  }
  ensureSelection(snapshot);
  renderSources(snapshot);
  renderSelectedExperiment(snapshot);
  renderSummary(snapshot);
  renderExperiments(snapshot);
  renderTrend(snapshot);
  renderDistributions(snapshot);
  renderModels(snapshot);
  renderRuns(snapshot);
  renderTraceInspector(snapshot);
}

function setTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === tabName));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === tabName));
}

function configureEvents() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setTab(tab.dataset.tab)));
  document.getElementById("refresh-button").addEventListener("click", loadSnapshot);
  document.getElementById("clear-experiment").addEventListener("click", () => {
    state.selectedExperimentKey = null;
    state.selectedTraceKey = null;
    state.selectedStepIndex = 0;
    render();
  });
  document.getElementById("trace-search").addEventListener("input", (event) => {
    state.traceQuery = event.target.value;
    state.selectedTraceKey = null;
    state.selectedStepIndex = 0;
    renderTraceInspector(state.snapshot);
  });
  document.getElementById("auto-refresh").addEventListener("change", (event) => {
    if (event.target.checked) startAutoRefresh();
    else stopAutoRefresh();
  });
}

function startAutoRefresh() {
  stopAutoRefresh();
  state.refreshTimer = setInterval(loadSnapshot, 3000);
}

function stopAutoRefresh() {
  if (state.refreshTimer) clearInterval(state.refreshTimer);
  state.refreshTimer = null;
}

configureEvents();
loadSnapshot();
startAutoRefresh();
