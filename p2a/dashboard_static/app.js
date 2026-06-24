const state = {
  snapshot: window.__P2A_DASHBOARD_SNAPSHOT__ || null,
  activeTab: "overview",
  selectedDataset: null,
  selectedEvalCellKey: null,
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
  return `${detail.eval_cell_key || detail.experiment_key || "cell"}::${detail.instance_id || detail.record_index}`;
}

function datasetRows(snapshot) {
  if (snapshot?.datasets?.length) return snapshot.datasets;
  const names = new Set();
  (snapshot?.eval_cells || snapshot?.experiments || []).forEach((row) => names.add(row.dataset || "unknown-dataset"));
  (snapshot?.details || []).forEach((row) => names.add(row.dataset || row.data_source || "unknown-dataset"));
  return [...names].sort().map((dataset) => ({ dataset }));
}

function experimentRows(snapshot) {
  return snapshot?.eval_cells || snapshot?.experiments || [];
}

function cellKey(row) {
  return row?.eval_cell_key || row?.experiment_key || "";
}

function detailCellKey(detail) {
  return detail?.eval_cell_key || detail?.experiment_key || "";
}

function selectedDatasetRow(snapshot) {
  return datasetRows(snapshot).find((row) => row.dataset === state.selectedDataset) || null;
}

function selectedExperiment(snapshot) {
  return experimentRows(snapshot).find((row) => cellKey(row) === state.selectedEvalCellKey) || null;
}

function ensureSelection(snapshot) {
  const datasets = datasetRows(snapshot);
  if (!datasets.length) {
    state.selectedDataset = null;
    state.selectedEvalCellKey = null;
    state.selectedExperimentKey = null;
    state.selectedTraceKey = null;
    return;
  }
  if (!state.selectedDataset || !datasets.some((row) => row.dataset === state.selectedDataset)) {
    state.selectedDataset = datasets.length === 1 ? datasets[0].dataset : null;
  }
  const cells = experimentRows(snapshot).filter((row) => !state.selectedDataset || row.dataset === state.selectedDataset);
  if (!state.selectedEvalCellKey && state.selectedExperimentKey) {
    state.selectedEvalCellKey = state.selectedExperimentKey;
  }
  if (!state.selectedEvalCellKey || !cells.some((row) => cellKey(row) === state.selectedEvalCellKey)) {
    state.selectedEvalCellKey = cells.length === 1 ? cellKey(cells[0]) : null;
  }
  state.selectedExperimentKey = state.selectedEvalCellKey;
  if (!state.selectedEvalCellKey) {
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
  const dataset = state.selectedDataset || "None";
  const cell = selected
    ? `${selected.source_kind || "-"} / ${selected.experiment_id || "-"} / ${selected.model_label || "-"}`
    : "None";
  const label = `Dataset: ${dataset} | Eval cell: ${cell}`;
  document.getElementById("selected-experiment").textContent = label;
}

function renderSummary(snapshot) {
  const counts = snapshot?.summary?.counts || {};
  const selectedDataset = selectedDatasetRow(snapshot);
  const cards = [
    ["Datasets", datasetRows(snapshot).length, fmt],
    ["Eval cells", experimentRows(snapshot).length, fmt],
    ["Dataset instances", selectedDataset?.n_instances ?? "-", fmt],
    ["Trajectories", selectedDataset?.n_trajectories ?? counts.n_records, fmt],
    ["Models", (snapshot.model_metrics || []).length, fmt],
    ["Runs", (snapshot.runs || []).length, fmt],
    ["Raw records", snapshot.raw_record_count ?? 0, fmt],
    ["Loaded details", snapshot.detail_count ?? counts.n_records, fmt],
  ];
  document.getElementById("summary-grid").innerHTML = cards.map(([label, value, formatter]) => metricCard(label, value, formatter)).join("");
}

function progress(row) {
  const done = row.done ?? row.detail_count ?? 0;
  const target = row.target ?? row.detail_count ?? 0;
  return `${done}/${target}`;
}

function renderExperiments(snapshot) {
  const datasetRowsHtml = datasetRows(snapshot).map((row) => {
    const selected = row.dataset === state.selectedDataset;
    return `<tr class="clickable ${selected ? "is-selected" : ""}" data-dataset="${esc(row.dataset)}">
      <td><button class="select-dataset" type="button" data-dataset="${esc(row.dataset)}">${selected ? "Selected" : "Select"}</button></td>
      <td>${esc(row.dataset)}</td>
      <td>${esc(row.n_instances ?? "-")}</td>
      <td>${esc(row.n_eval_cells ?? "-")}</td>
      <td>${esc(row.n_trajectories ?? "-")}</td>
      <td>${esc((row.models || []).join(", ") || "-")}</td>
      <td>${esc((row.source_kinds || []).join(", ") || "-")}</td>
    </tr>`;
  });
  const cellRows = experimentRows(snapshot).filter((row) => !state.selectedDataset || row.dataset === state.selectedDataset);
  const rows = cellRows.map((row) => {
    const key = cellKey(row);
    const selected = key === state.selectedEvalCellKey;
    return `<tr class="clickable ${selected ? "is-selected" : ""}" data-eval-cell-key="${esc(key)}">
      <td><button class="select-cell" type="button" data-eval-cell-key="${esc(key)}">${selected ? "Selected" : "Inspect"}</button></td>
      <td>${esc(row.source_kind)}</td>
      <td>${esc(row.experiment_id)}</td>
      <td>${esc(row.provider_source)}</td>
      <td>${esc(row.dataset)}</td>
      <td>${esc(row.model_label)}</td>
      <td>${esc(progress(row))}</td>
      <td>${esc(row.trajectory_count ?? 0)}</td>
    </tr>`;
  });
  document.getElementById("experiment-table").innerHTML = `
    <section class="subsection"><h3>Datasets</h3>${table(["", "Dataset", "Instances", "Eval cells", "Trajectories", "Models", "Sources"], datasetRowsHtml)}</section>
    <section class="subsection"><h3>Eval cells${state.selectedDataset ? ` in ${esc(state.selectedDataset)}` : ""}</h3>${table(["", "Kind", "Experiment", "Provider", "Dataset", "Model", "Done", "Traj"], rows)}</section>`;
  document.querySelectorAll(".select-dataset, #experiment-table tr[data-dataset]").forEach((el) => {
    el.addEventListener("click", () => {
      const dataset = el.dataset.dataset;
      if (!dataset) return;
      state.selectedDataset = dataset;
      state.selectedEvalCellKey = null;
      state.selectedExperimentKey = null;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      render();
    });
  });
  document.querySelectorAll(".select-cell, #experiment-table tr[data-eval-cell-key]").forEach((el) => {
    el.addEventListener("click", () => {
      const key = el.dataset.evalCellKey;
      if (!key) return;
      state.selectedEvalCellKey = key;
      state.selectedExperimentKey = key;
      const cell = experimentRows(state.snapshot).find((row) => cellKey(row) === key);
      if (cell?.dataset) state.selectedDataset = cell.dataset;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      setTab(el.classList.contains("select-cell") ? "traces" : state.activeTab);
      render();
    });
  });
}

function scopedRows(rows) {
  if (!state.selectedEvalCellKey) return [];
  return (rows || []).filter((row) => cellKey(row) === state.selectedEvalCellKey);
}

function renderTrend(snapshot) {
  const dataset = state.selectedDataset;
  const trends = snapshot?.summary?.trends || [];
  const rows = trends
    .filter((row) => !dataset || row.data_source === dataset)
    .map((row) => {
      const rates = row.rates || {};
      return `<tr>
        <td>${esc(row.data_source)}</td>
        <td>${esc(row.run_step)}</td>
        <td>${esc(row.n_records)}</td>
        <td>${esc(pct(rates.bonus_map_coverage))}</td>
        <td>${esc(pct(rates.call_graph_coverage))}</td>
        <td>${esc(pct(rates.read_rate))}</td>
        <td>${esc(pct(rates.chain_graph_coverage))}</td>
      </tr>`;
    });
  document.getElementById("trend-table").innerHTML = table(
    ["Data source", "Step", "N", "Bonus maps", "Call graphs", "Read rate", "Chain coverage"],
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
  if (!state.selectedDataset) {
    document.getElementById("distribution-grid").innerHTML = '<div class="empty">Select a dataset before inspecting distributions.</div>';
    return;
  }
  const payload = snapshot?.summary?.distributions_by_dataset?.[state.selectedDataset] || snapshot?.summary?.by_dataset?.[state.selectedDataset] || {};
  const dist = payload.distributions || {};
  const population = payload.n_instances || payload.counts?.n_instances || 0;
  document.getElementById("distribution-grid").innerHTML = [
    miniTable(`Dataset population (${state.selectedDataset})`, { unique_instances: population }),
    miniTable("Case types", dist.case_types),
    miniTable("Not chain-evaluable", dist.not_chain_evaluable_reasons),
    miniTable("Graph availability", dist.availability),
  ].join("");
}

function renderModels(snapshot) {
  if (!state.selectedDataset) {
    document.getElementById("model-table").innerHTML = '<div class="empty">Select a dataset in Overview before comparing Macro KPIs.</div>';
    return;
  }
  const rows = (snapshot?.model_metrics || []).filter((row) => row.dataset === state.selectedDataset);
  const hasCacheWrite = rows.some((row) => row.cache_write_rate !== null && row.cache_write_rate !== undefined);
  const headers = [
    "",
    "Model",
    "Kind",
    "Experiment",
    "Done",
    "Task success",
    "Read precision",
    "Node recall",
    "F1",
    "Anchor",
    "Root",
    "Chain recall",
    "Chain precision",
    "First anchor",
    "First root",
    "Order",
    "Reverse",
    "Miracle",
    "Blocks",
    "Achieved",
    "Wasted",
    "Loop blocks",
    "Loop trace",
    "Error spiral",
    "Turns",
    "Tools",
    "Wall",
    "In",
    "Out",
    "Reason",
    "Cost",
    "Cache hit",
  ];
  if (hasCacheWrite) headers.push("Cache write");
  const tableRows = rows.map((row) => {
    const key = cellKey(row);
    const cells = [
      `<button class="select-kpi-cell" type="button" data-eval-cell-key="${esc(key)}">${key === state.selectedEvalCellKey ? "Selected" : "Select"}</button>`,
      row.model_label,
      row.source_kind,
      row.experiment_id,
      progress(row),
      pct(row.resolved_rate ?? row.reward_rate),
      pct(row.avg_chain_read_precision ?? row.avg_read_precision),
      pct(row.avg_chain_node_recall ?? row.avg_node_recall),
      pct(row.avg_hit_f1),
      pct(row.anchor_hit_rate),
      pct(row.root_hit_rate),
      pct(row.avg_chain_node_recall),
      pct(row.avg_chain_read_precision),
      fmt(row.avg_first_anchor_step, 1),
      fmt(row.avg_first_root_step, 1),
      fmt(row.avg_order_score),
      pct(row.reverse_order_rate),
      pct(row.miracle_rate),
      fmt(row.avg_blocks_per_trace, 1),
      pct(row.block_achieve_rate),
      pct(row.block_waste_rate),
      pct(row.block_loop_rate),
      pct(row.loop_trace_rate),
      pct(row.error_spiral_rate),
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
    return `<tr class="${key === state.selectedEvalCellKey ? "is-selected" : ""}" data-eval-cell-key="${esc(key)}">${cells.map((cell, index) => `<td>${index === 0 ? cell : esc(cell)}</td>`).join("")}</tr>`;
  });
  document.getElementById("model-table").innerHTML = `
    <div class="panel-note">Macro KPI is scoped to dataset <strong>${esc(state.selectedDataset)}</strong>. Effect and dependency-rigor metrics are placed before efficiency/cost metrics.</div>
    <div class="table-wrap kpi-table">${table(headers, tableRows)}</div>`;
  document.querySelectorAll(".select-kpi-cell, #model-table tr[data-eval-cell-key]").forEach((el) => {
    el.addEventListener("click", () => {
      const key = el.dataset.evalCellKey;
      if (!key) return;
      state.selectedEvalCellKey = key;
      state.selectedExperimentKey = key;
      const cell = experimentRows(state.snapshot).find((row) => cellKey(row) === key);
      if (cell?.dataset) state.selectedDataset = cell.dataset;
      state.selectedTraceKey = null;
      state.selectedStepIndex = 0;
      render();
    });
  });
}

function renderRuns(snapshot) {
  const selected = selectedExperiment(snapshot);
  if (!state.selectedDataset) {
    document.getElementById("run-list").innerHTML = '<div class="empty">Select a dataset before inspecting run provenance.</div>';
    return;
  }
  const allRuns = snapshot?.runs || [];
  const linked = allRuns.filter((run) => {
    const keys = run.eval_cell_keys || [];
    if (selected) return keys.includes(cellKey(selected));
    return (run.datasets || []).includes(state.selectedDataset);
  });
  const unlinked = allRuns.filter((run) => !(run.eval_cell_keys || []).length);
  const renderCards = (runs) => runs.map((run) => {
    const statusTone = run.status === "completed" ? "ok" : run.status === "running" || run.status === "verify" ? "warn" : "";
    const files = (run.files || []).slice(0, 8).map((name) => `<span class="badge">${esc(name)}</span>`).join("");
    const links = (run.eval_cell_keys || []).length
      ? `<div class="run-meta">Linked cells: ${esc((run.model_labels || []).join(", ") || run.eval_cell_keys.length)}</div>`
      : '<div class="run-meta">Unlinked provenance: no eval-cell metadata in this artifact.</div>';
    const log = run.log_excerpt ? `<pre class="log">${esc(run.log_excerpt.slice(-6000))}</pre>` : '<div class="muted">No run.log tail.</div>';
    return `<article class="run-card">
      <div class="run-head"><div><div class="run-title">${esc(run.run_id)}</div><div class="run-meta">${esc(run.path)}</div></div>${badge(run.status || "unknown", true, statusTone)}</div>
      <div class="run-meta">Updated ${run.last_update ? new Date(run.last_update * 1000).toLocaleString() : "-"}</div>${links}
      <div>${files}</div>${log}
    </article>`;
  }).join("");
  document.getElementById("run-list").innerHTML = `
    <div class="panel-note">Run Provenance shows artifact-producing executions. Quality metrics live in Macro KPI; trajectories live in Trajectories.</div>
    <h3>${selected ? "Linked to selected eval cell" : `Linked to dataset ${esc(state.selectedDataset)}`}</h3>
    <div class="run-grid">${renderCards(linked) || '<div class="empty">No explicitly linked runs for this scope.</div>'}</div>
    <h3>Unlinked runs</h3>
    <div class="run-grid">${renderCards(unlinked) || '<div class="empty">No unlinked runs.</div>'}</div>`;
}

function traceBlob(detail) {
  return JSON.stringify({
    instance_id: detail.instance_id,
    files: detail.read_files,
    step_reads: (detail.step_inspection || []).map((step) => step.recovered_reads || step.target_path || step.path),
    bad: detail.bad_patterns,
    chain_bad: detail.chain_bad_patterns,
    reason: detail.not_chain_evaluable_reason,
  }).toLowerCase();
}

function filteredDetails(snapshot) {
  const query = state.traceQuery.trim().toLowerCase();
  if (!state.selectedEvalCellKey) return [];
  return (snapshot?.details || [])
    .filter((detail) => detailCellKey(detail) === state.selectedEvalCellKey)
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
  if ((scored.n_reads || 0) > 0 || (step?.recovered_reads || []).length || step?.action_family === "read") return "offmap";
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
  const layers = new Map();
  nodes.forEach((node) => {
    const raw = node.normalized_distance;
    const distance = typeof raw === "number" ? raw : 1;
    const layer = Math.max(0, Math.min(5, Math.round(distance * 5)));
    if (!layers.has(layer)) layers.set(layer, []);
    layers.get(layer).push(node);
  });
  const sortedLayers = [...layers.entries()].sort((a, b) => b[0] - a[0]);
  const maxLayerSize = Math.max(...sortedLayers.map((entry) => entry[1].length), 1);
  const rowGap = 70;
  const colGap = 190;
  const width = Math.max(760, 220 + sortedLayers.length * colGap);
  const height = Math.max(260, 90 + maxLayerSize * rowGap);
  const positions = new Map();
  sortedLayers.forEach(([_layer, layerNodes], layerIndex) => {
    const x = 80 + layerIndex * colGap;
    const sortedNodes = [...layerNodes].sort((a, b) => String(a.key).localeCompare(String(b.key)));
    sortedNodes.forEach((node, index) => positions.set(node.key, { x, y: 55 + index * rowGap }));
  });
  const edgeSvg = edges.map(([caller, callee]) => {
    const a = positions.get(caller);
    const b = positions.get(callee);
    if (!a || !b) return "";
    return `<line class="graph-edge" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"><title>${esc(caller)} -> ${esc(callee)}</title></line>`;
  }).join("");
  const nodeSvg = nodes.map((node) => {
    const pos = positions.get(node.key);
    if (!pos) return "";
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
  const tool = step.tool_name || (step.tool_names || [step.scored?.family || "step"]).join("+");
  const family = step.action_family && step.action_family !== "other" ? `${step.action_family}: ` : "";
  const target = step.target_path || step.path || step.scored?.target_path || "";
  return `<button type="button" class="step-thumb ${tone} ${selected ? "is-selected" : ""}" data-step-index="${traceIndex}">
    <span class="step-num">${esc(step.step_index ?? traceIndex)}</span>
    <span class="step-tool">${esc(`${family}${tool}`)}</span>
    <span class="step-target">${esc(target.split("/").slice(-2).join("/") || step.command || "no target")}</span>
  </button>`;
}

function jsonBlock(value) {
  const text = typeof value === "string" ? value : JSON.stringify(value ?? "", null, 2);
  return `<pre class="detail-pre">${esc(text || "-")}</pre>`;
}

function commandLine(step) {
  const parts = [step.tool_name || (step.tool_names || []).join("+") || "step"];
  if (step.command) parts.push(step.command);
  if (step.path) parts.push(step.path);
  if (step.view_range) parts.push(Array.isArray(step.view_range) ? `[${step.view_range.join(", ")}]` : step.view_range);
  return parts.filter(Boolean).join(" ");
}

function renderReadList(reads) {
  const rows = (reads || []).map((read) => `<tr><td>${esc(read.file_path || read.path)}</td><td>${esc(read.start_line ?? "-")}</td><td>${esc(read.end_line ?? "-")}</td></tr>`);
  return table(["File", "Start", "End"], rows);
}

function renderDiff(oldText, newText) {
  if (oldText === undefined && newText === undefined) return "";
  const oldLines = String(oldText || "").split("\n");
  const newLines = String(newText || "").split("\n");
  const rows = [];
  const maxLen = Math.max(oldLines.length, newLines.length);
  for (let i = 0; i < maxLen; i += 1) {
    if (oldLines[i] !== undefined && oldLines[i] !== newLines[i]) {
      rows.push(`<div class="diff-line diff-del"><span>-</span><code>${esc(oldLines[i])}</code></div>`);
    }
    if (newLines[i] !== undefined && oldLines[i] !== newLines[i]) {
      rows.push(`<div class="diff-line diff-add"><span>+</span><code>${esc(newLines[i])}</code></div>`);
    }
    if (oldLines[i] !== undefined && oldLines[i] === newLines[i]) {
      rows.push(`<div class="diff-line diff-ctx"><span> </span><code>${esc(oldLines[i])}</code></div>`);
    }
  }
  return `<section><h4>Inline diff</h4><div class="diff-view">${rows.join("") || '<div class="muted">No textual difference.</div>'}</div></section>`;
}

function renderToolCallSummary(step) {
  const args = (step.tool_args || [])[0] || {};
  const fields = {
    tool: step.tool_name || (step.tool_names || []).join("+") || "-",
    family: step.action_family || "-",
    command: step.command || args.command || "-",
    path: step.path || step.target_path || args.path || args.file || "-",
    view_range: step.view_range || args.view_range || "-",
    status: step.status || "-",
  };
  return table(["Field", "Value"], Object.entries(fields).map(([key, value]) => `<tr><td>${esc(key)}</td><td>${esc(Array.isArray(value) ? value.join(", ") : value)}</td></tr>`));
}

function renderStepDetail(detail) {
  const byIndex = stepByTraceIndex(detail);
  const step = byIndex.get(Number(state.selectedStepIndex)) || [...byIndex.values()][0];
  if (!step) {
    return `<section class="step-detail"><h3>Trajectory detail</h3><div class="empty">Raw step content was not captured for this artifact.</div></section>`;
  }
  const scored = step.scored || {};
  const hitNodes = (scored.hit_nodes || []).map((node) => `${node.node_role || "node"}: ${node.key}`).join("\n");
  const reads = scored.reads || step.recovered_reads || [];
  return `<section class="step-detail">
    <h3>Step ${esc(step.step_index ?? step.trace_index)}</h3>
    <div class="detail-badges">
      ${badge(commandLine(step))}
      ${step.parse_error ? badge("parse error", true, "bad") : ""}
      ${step.exit_reason ? badge(step.exit_reason) : ""}
      ${reads.length ? badge(`${reads.length} recovered reads`, true, "ok") : badge("no recovered read", true, "warn")}
    </div>
    <div class="detail-grid">
      <section><h4>Tool summary</h4>${renderToolCallSummary(step)}</section>
      <section><h4>Think / assistant text</h4>${jsonBlock(step.thought || step.response_text || "(empty)")}</section>
      <section><h4>Action</h4>${jsonBlock(step.raw_action || step.tool_calls || [])}</section>
      ${renderDiff(step.old_str, step.new_str)}
      <section><h4>Observation</h4>${jsonBlock(step.observation || step.tool_results || [])}</section>
      <section><h4>Recovered reads</h4>${renderReadList(reads)}</section>
      <section><h4>Matched bonus-map nodes</h4>${jsonBlock(hitNodes || "No matched bonus-map node.")}</section>
    </div>
  </section>`;
}

function renderTraceInspector(snapshot) {
  const detail = selectedDetail(snapshot);
  if (!state.selectedEvalCellKey) {
    document.getElementById("trace-inspector").innerHTML = '<div class="empty">Select a dataset and eval cell/model before inspecting trajectories.</div>';
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
    state.selectedDataset = null;
    state.selectedEvalCellKey = null;
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
