const state = {
  snapshot: window.__P2A_DASHBOARD_SNAPSHOT__ || null,
  activeTab: "overview",
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

function badge(label, active, tone = "ok") {
  return `<span class="badge ${active ? tone : ""}">${esc(label)}</span>`;
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

function renderSummary(snapshot) {
  const summary = snapshot?.summary || {};
  const rates = summary.rates || {};
  const avg = summary.averages || {};
  const counts = summary.counts || {};
  const cards = [
    ["Records", counts.n_records, fmt],
    ["Models", (snapshot.model_metrics || []).length, fmt],
    ["Runs", (snapshot.runs || []).length, fmt],
    ["Chain coverage", rates.chain_graph_coverage, pct],
    ["Anchor hit", rates.anchor_hit_rate, pct],
    ["Root hit", rates.root_hit_rate, pct],
    ["Chain hit", rates.chain_hit_rate, pct],
    ["Chain recall", rates.chain_node_recall, pct],
    ["Chain precision", rates.chain_read_precision, pct],
    ["Time to anchor", avg.time_to_anchor, fmt],
    ["Time to root", avg.time_to_root, fmt],
    ["Not evaluable", counts.n_not_chain_evaluable, fmt],
  ];
  document.getElementById("summary-grid").innerHTML = cards.map(([label, value, formatter]) => metricCard(label, value, formatter)).join("");
}

function table(headers, rows) {
  if (!rows.length) return '<div class="empty">No rows.</div>';
  return `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

function renderTrend(snapshot) {
  const trends = snapshot?.summary?.trends || [];
  const rows = trends.map((row) => {
    const rates = row.rates || {};
    const avg = row.averages || {};
    return `<tr>
      <td>${esc(row.data_source)}</td>
      <td>${esc(row.run_step)}</td>
      <td>${esc(row.n_records)}</td>
      <td>${esc(pct(rates.chain_graph_coverage))}</td>
      <td>${esc(pct(rates.anchor_hit_rate))}</td>
      <td>${esc(pct(rates.root_hit_rate))}</td>
      <td>${esc(pct(rates.chain_hit_rate))}</td>
      <td>${esc(pct(rates.chain_node_recall))}</td>
      <td>${esc(fmt(avg.steps_anchor_to_root))}</td>
      <td>${esc(pct(rates.anchor_before_root_rate))}</td>
    </tr>`;
  });
  document.getElementById("trend-table").innerHTML = table(
    ["Data source", "Step", "N", "Coverage", "Anchor", "Root", "Chain", "Recall", "Anchor->root", "Order"],
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

function progress(row) {
  const done = row.done || 0;
  const target = row.target || 0;
  return `${done}/${target}`;
}

function modelFlags(row) {
  const flags = [];
  if (row.errors) flags.push(badge(`${row.errors} errors`, true, "bad"));
  if (row.pending) flags.push(badge(`${row.pending} pending`, true, "warn"));
  if (row.p2a_read_rate === 0) flags.push(badge("empty read", true, "bad"));
  if ((row.avg_turns || 0) >= 16) flags.push(badge("near cap", true, "warn"));
  return flags.join(" ");
}

function renderModels(snapshot) {
  const rows = (snapshot?.model_metrics || []).map((row) => `<tr>
    <td>${esc(row.model_label)}</td>
    <td>${esc(row.provider_source)}</td>
    <td>${esc(row.dataset)}</td>
    <td>${esc(progress(row))}</td>
    <td>${esc(pct(row.resolved_rate))}</td>
    <td>${esc(pct(row.reward_rate))}</td>
    <td>${esc(pct(row.p2a_read_rate))}</td>
    <td>${esc(pct(row.call_graph_hit_rate))}</td>
    <td>${esc(pct(row.ground_truth_hit_rate))}</td>
    <td>${esc(pct(row.near_hit_rate))}</td>
    <td>${esc(fmt(row.avg_min_distance))}</td>
    <td>${esc(fmt(row.avg_turns, 1))}</td>
    <td>${esc(fmt(row.avg_tool_calls, 1))}</td>
    <td>${esc(fmt(row.avg_wall_time, 1))}</td>
    <td>${esc(token(row.avg_input_tokens))}</td>
    <td>${esc(token(row.avg_output_tokens))}</td>
    <td>${esc(token(row.avg_reasoning_tokens))}</td>
    <td>${esc(pct(row.cache_hit_rate))}</td>
    <td>${esc(pct(row.cache_write_rate))}</td>
    <td>${modelFlags(row)}</td>
  </tr>`);
  document.getElementById("model-table").innerHTML = table(
    ["Model", "Provider", "Dataset", "Done", "Resolved", "Reward", "P2A read", "Call graph", "Root", "Near", "Distance", "Turns", "Tools", "Wall", "In", "Out", "Reason", "Cache hit", "Cache write", "Flags"],
    rows
  );
}

function renderRuns(snapshot) {
  const runs = snapshot?.runs || [];
  const cards = runs.map((run) => {
    const statusTone = run.status === "completed" ? "ok" : run.status === "running" || run.status === "verify" ? "warn" : "";
    const files = (run.files || []).slice(0, 8).map((name) => `<span class="badge">${esc(name)}</span>`).join("");
    const log = run.log_excerpt ? `<pre class="log">${esc(run.log_excerpt.slice(-6000))}</pre>` : '<div class="muted">No run.log tail.</div>';
    return `<article class="run-card">
      <div class="run-head">
        <div>
          <div class="run-title">${esc(run.run_id)}</div>
          <div class="run-meta">${esc(run.path)}</div>
        </div>
        ${badge(run.status || "unknown", true, statusTone)}
      </div>
      <div class="run-meta">Updated ${run.last_update ? new Date(run.last_update * 1000).toLocaleString() : "-"}</div>
      <div>${files}</div>
      ${log}
    </article>`;
  });
  document.getElementById("run-list").innerHTML = cards.join("") || '<div class="empty">No run directories discovered.</div>';
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

function renderNodeTable(nodes) {
  const rows = (nodes || []).map((node) => `<tr class="${node.group === "context" ? "context" : "chain"}">
    <td>${esc(node.key)}</td>
    <td>${esc(node.file_path)}:${esc(node.start_line)}-${esc(node.end_line)}</td>
    <td>${esc(node.node_role)}</td>
    <td>${badge("hit", Boolean(node.hit), "ok")}</td>
    <td>${esc(fmt(node.first_step))}</td>
  </tr>${node.source_preview ? `<tr><td colspan="5"><pre class="source-preview">${esc(node.source_preview)}</pre></td></tr>` : ""}`);
  return `<div class="table-wrap"><table class="node-table"><thead><tr><th>Node</th><th>Range</th><th>Role</th><th>Hit</th><th>First step</th></tr></thead><tbody>${rows.join("") || '<tr><td colspan="5">No nodes.</td></tr>'}</tbody></table></div>`;
}

function renderEdges(title, edges) {
  const items = (edges || []).map((edge) => `<li>${esc(edge.caller)} -> ${esc(edge.callee)} <span class="muted">${esc(edge.role_transition)}</span></li>`);
  return `<section class="mini-panel"><h3>${esc(title)}</h3><ul class="edge-list">${items.join("") || "<li>No edges.</li>"}</ul></section>`;
}

function renderBlocks(blocks) {
  if (!blocks || !blocks.length) return '<div class="muted">No purpose blocks.</div>';
  return blocks.slice(0, 80).map((block) => {
    const stateName = block.achieved ? "achieved" : block.loop ? "loop" : block.wasted ? "wasted" : "neutral";
    const tone = block.achieved ? "ok" : block.loop || block.wasted ? "bad" : "";
    return `<div class="block-line">${badge(stateName, true, tone)}
      <strong>Block ${esc(block.block_index)}</strong>
      ${esc(block.family)} ${esc(block.target_path)}
      steps ${esc(JSON.stringify(block.step_indices || []))}
      first hit ${esc(fmt(block.first_hit_step))}
      d=${esc(fmt(block.min_distance))}
    </div>`;
  }).join("");
}

function renderSteps(steps) {
  const rows = (steps || []).slice(0, 120).map((step) => {
    const hits = (step.hit_nodes || []).map((node) => node.key).join(", ");
    return `<tr>
      <td>${esc(step.step_index)}</td>
      <td>${esc(step.family)}</td>
      <td>${esc(step.target_path)}</td>
      <td>${esc(step.n_reads)}</td>
      <td>${esc(fmt(step.min_distance))}</td>
      <td>${esc(hits || "-")}</td>
    </tr>`;
  });
  return table(["Step", "Family", "Target", "Reads", "Distance", "Matched nodes"], rows);
}

function renderTraces(snapshot) {
  const query = state.traceQuery.trim().toLowerCase();
  const details = (snapshot?.details || []).filter((detail) => !query || traceBlob(detail).includes(query));
  const cards = details.map((detail) => {
    const projection = detail.chain_projection || {};
    const nodes = [...(projection.context_nodes || []), ...(projection.chain_nodes || [])];
    const chainBad = detail.chain_bad_patterns || {};
    const bad = detail.bad_patterns || {};
    const badges = [
      badge("chain", Boolean(detail.chain_hit), "ok"),
      badge("anchor", Boolean(detail.anchor_hit), "ok"),
      badge("root", Boolean(detail.root_hit), "ok"),
      badge("evaluable", Boolean(detail.chain_evaluable), "ok"),
      badge("loop", Boolean(bad.has_loop), "bad"),
      badge("error spiral", Boolean(bad.error_spiral), "bad"),
      badge("missed anchor", Boolean(chainBad.missed_anchor), "bad"),
      badge("missed root", Boolean(chainBad.missed_root_after_anchor), "bad"),
    ].join("");
    return `<details class="trace-card">
      <summary>
        <div class="trace-head">
          <div>
            <div class="trace-title">${esc(detail.instance_id || `record-${detail.record_index}`)}</div>
            <div class="run-meta">chain recall ${esc(pct(detail.chain_node_recall))} | precision ${esc(pct(detail.chain_read_precision))} | anchor ${esc(fmt(detail.first_anchor_step))} | root ${esc(fmt(detail.first_root_step))}</div>
          </div>
          <div>${badges}</div>
        </div>
      </summary>
      <div class="trace-body">
        <h3>Dependency graph projection</h3>
        ${detail.chain_evaluable ? "" : `<div class="muted">Not chain-evaluable: ${esc(detail.not_chain_evaluable_reason)}</div>`}
        ${renderNodeTable(nodes)}
        <div class="split-grid">${renderEdges("Chain edges", projection.chain_edges)}${renderEdges("Context edges", projection.context_edges)}</div>
        <h3>Purpose blocks</h3>
        ${renderBlocks(detail.purpose_blocks)}
        <h3>Step annotations</h3>
        <div class="table-wrap">${renderSteps(detail.step_details)}</div>
      </div>
    </details>`;
  });
  document.getElementById("trace-list").innerHTML = cards.join("") || '<div class="empty">No trajectories match this view.</div>';
}

function render() {
  const snapshot = state.snapshot;
  if (!snapshot) {
    document.getElementById("summary-grid").innerHTML = '<div class="empty">Dashboard data is not available.</div>';
    return;
  }
  renderSources(snapshot);
  renderSummary(snapshot);
  renderTrend(snapshot);
  renderDistributions(snapshot);
  renderModels(snapshot);
  renderRuns(snapshot);
  renderTraces(snapshot);
}

function setTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === tabName));
  document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === tabName));
}

function configureEvents() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setTab(tab.dataset.tab)));
  document.getElementById("refresh-button").addEventListener("click", loadSnapshot);
  document.getElementById("trace-search").addEventListener("input", (event) => {
    state.traceQuery = event.target.value;
    renderTraces(state.snapshot);
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
