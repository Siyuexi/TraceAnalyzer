import json
import subprocess
import textwrap
from pathlib import Path


def test_dashboard_frontend_state_and_inspector_rendering(tmp_path):
    app_path = Path(__file__).resolve().parents[1] / "p2a" / "dashboard_static" / "app.js"
    snapshot = {
        "schema_version": "p2a_unified_dashboard_v1",
        "sources": [{"kind": "db", "path": "demo.sqlite"}],
        "summary": {
            "counts": {"n_records": 2, "n_not_chain_evaluable": 0},
            "rates": {"anchor_hit_rate": 1.0, "root_hit_rate": 1.0, "chain_node_recall": 1.0},
            "averages": {},
            "distributions": {},
            "distributions_by_dataset": {
                "swebench-hard": {
                    "dataset": "swebench-hard",
                    "n_instances": 2,
                    "distributions": {
                        "case_types": {"direct": 2},
                        "not_chain_evaluable_reasons": {},
                        "availability": {"with_bonus_map": 2, "with_call_graph": 2, "chain_evaluable": 2, "not_chain_evaluable": 0},
                    },
                }
            },
            "by_case_type": {},
            "trends": [],
        },
        "datasets": [
            {
                "dataset": "swebench-hard",
                "n_instances": 2,
                "n_eval_cells": 2,
                "n_trajectories": 2,
                "models": ["model-a", "model-b"],
                "source_kinds": ["third_party_api"],
            }
        ],
        "eval_cells": [
            {
                "eval_cell_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "source_kind": "third_party_api",
                "experiment_id": "exp-a",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_api_name": "model-a",
                "model_label": "model-a",
                "target": 1,
                "done": 1,
                "trajectory_count": 1,
                "resolved_rate": 0.0,
                "root_hit_rate": 1.0,
                "chain_node_recall": 1.0,
                "read_precision": 1.0,
            },
            {
                "eval_cell_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "experiment_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "source_kind": "third_party_api",
                "experiment_id": "exp-b",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_api_name": "model-b",
                "model_label": "model-b",
                "target": 1,
                "done": 1,
                "trajectory_count": 1,
            },
        ],
        "experiments": [],
        "model_metrics": [],
        "runs": [],
        "details": [
            {
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "eval_cell_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
                "experiment_id": "exp-a",
                "source_kind": "third_party_api",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_label": "model-a",
                "instance_id": "case-a",
                "record_index": 0,
                "run_id": "run-a",
                "root_hit": True,
                "anchor_hit": True,
                "chain_hit": True,
                "chain_evaluable": True,
                "chain_node_recall": 1.0,
                "chain_read_precision": 1.0,
                "bad_patterns": {"has_loop": False, "error_spiral": False},
                "chain_bad_patterns": {},
                "chain_projection": {
                    "anchors": ["pkg/symptom.py::symptom"],
                    "roots": ["pkg/root.py::root"],
                    "context_nodes": [],
                    "chain_nodes": [
                        {
                            "key": "pkg/symptom.py::symptom",
                            "file_path": "pkg/symptom.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 1.0,
                            "node_role": "symptom",
                            "hit": True,
                            "first_step": 0,
                        },
                        {
                            "key": "pkg/root.py::root",
                            "file_path": "pkg/root.py",
                            "start_line": 1,
                            "end_line": 10,
                            "normalized_distance": 0.0,
                            "node_role": "root_cause",
                            "hit": True,
                            "first_step": 1,
                        },
                    ],
                    "chain_edges": [
                        {"caller": "pkg/symptom.py::symptom", "callee": "pkg/root.py::root"}
                    ],
                    "context_edges": [],
                },
                "purpose_blocks": [
                    {
                        "block_index": 0,
                        "family": "read",
                        "target_path": "pkg/root.py",
                        "step_indices": [1, 2],
                        "trace_indices": [0, 1],
                        "achieved": True,
                        "wasted": False,
                        "loop": False,
                    }
                ],
                "step_inspection": [
                    {
                        "trace_index": 0,
                        "step_index": 0,
                        "tool_names": ["str_replace_editor"],
                        "tool_name": "str_replace_editor",
                        "action_family": "read",
                        "command": "view",
                        "path": "/testbed/pkg/symptom.py",
                        "view_range": [1, 10],
                        "thought": "inspect symptom",
                        "response_text": "view symptom",
                        "tool_args": [{"command": "view", "path": "/testbed/pkg/symptom.py", "view_range": [1, 10]}],
                        "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed/pkg/symptom.py", "view_range": [1, 10]}}}],
                        "tool_results": [{"observation": "symptom body"}],
                        "observation": "symptom body",
                        "recovered_reads": [{"file_path": "pkg/symptom.py", "start_line": 1, "end_line": 10}],
                        "scored": {
                            "trace_index": 0,
                            "step_index": 0,
                            "target_path": "pkg/symptom.py",
                            "n_reads": 1,
                            "reads": [{"file_path": "pkg/symptom.py", "start_line": 1, "end_line": 10}],
                            "hit_nodes": [{"key": "pkg/symptom.py::symptom", "node_role": "symptom"}],
                        },
                    },
                    {
                        "trace_index": 1,
                        "step_index": 1,
                        "tool_names": ["str_replace_editor"],
                        "tool_name": "str_replace_editor",
                        "action_family": "edit",
                        "command": "str_replace",
                        "path": "/testbed/pkg/root.py",
                        "old_str": "return bad",
                        "new_str": "return good",
                        "thought": "inspect root",
                        "response_text": "view root",
                        "tool_args": [{"command": "str_replace", "path": "/testbed/pkg/root.py", "old_str": "return bad", "new_str": "return good"}],
                        "tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "str_replace", "path": "/testbed/pkg/root.py"}}}],
                        "tool_results": [{"observation": "root body"}],
                        "observation": "root body",
                        "recovered_reads": [{"file_path": "pkg/root.py", "start_line": 1, "end_line": 10}],
                        "scored": {
                            "trace_index": 1,
                            "step_index": 1,
                            "target_path": "pkg/root.py",
                            "n_reads": 1,
                            "reads": [{"file_path": "pkg/root.py", "start_line": 1, "end_line": 10}],
                            "hit_nodes": [{"key": "pkg/root.py::root", "node_role": "root_cause"}],
                        },
                    },
                ],
            },
            {
                "experiment_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "eval_cell_key": "third_party_api::exp-b::internal_api::swebench-hard::model-b::model-b",
                "experiment_id": "exp-b",
                "source_kind": "third_party_api",
                "provider_source": "internal_api",
                "dataset": "swebench-hard",
                "model_label": "model-b",
                "instance_id": "case-b",
                "record_index": 1,
                "step_inspection": [],
            },
        ],
    }
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    harness = tmp_path / "frontend_smoke.cjs"
    harness.write_text(
        textwrap.dedent(
            """
            const fs = require("fs");
            const vm = require("vm");
            const appPath = process.argv[2];
            const snapshot = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
            class Element {
              constructor(id) {
                this.id = id;
                this.innerHTML = "";
                this.textContent = "";
                this.dataset = {};
                this.classList = { toggle() {}, add() {}, remove() {}, contains() { return false; } };
              }
              addEventListener() {}
            }
            const elements = new Map();
            const document = {
              getElementById(id) {
                if (!elements.has(id)) elements.set(id, new Element(id));
                return elements.get(id);
              },
              querySelectorAll() { return []; },
            };
            const context = {
              window: { __P2A_DASHBOARD_SNAPSHOT__: snapshot },
              document,
              console,
              fetch: async () => { throw new Error("network disabled"); },
              setInterval: () => 1,
              clearInterval: () => {},
            };
            vm.createContext(context);
            vm.runInContext(fs.readFileSync(appPath, "utf8"), context);
            function run(expr) { return vm.runInContext(expr, context); }
            if (run("state.selectedDataset") !== "swebench-hard") {
              throw new Error("single dataset should be auto-selected");
            }
            if (run("state.selectedEvalCellKey") !== null || run("state.selectedExperimentKey") !== null) {
              throw new Error("multi-cell dataset must require explicit model/cell selection");
            }
            const expHtml = elements.get("experiment-table").innerHTML;
            if (!expHtml.includes("Datasets") || !expHtml.includes("Eval cells") || !expHtml.includes("exp-a") || !expHtml.includes("exp-b")) {
              throw new Error("dataset/eval-cell registry did not render");
            }
            const firstCell = snapshot.eval_cells[0].eval_cell_key;
            run(`state.selectedEvalCellKey = ${JSON.stringify("PLACEHOLDER")};
                 state.selectedExperimentKey = ${JSON.stringify("PLACEHOLDER")};
                 state.selectedTraceKey = ${JSON.stringify("TRACEKEY")};
                 state.selectedStepIndex = 1;
                 render();`.replaceAll("PLACEHOLDER", firstCell).replaceAll("TRACEKEY", `${firstCell}::case-a`));
            const traceHtml = elements.get("trace-inspector").innerHTML;
            for (const needle of ["trace-left", "trace-middle", "trace-right", "<svg", "Purpose blocks", "Tool summary", "Observation", "Inline diff", "Matched bonus-map nodes"]) {
              if (!traceHtml.includes(needle)) throw new Error(`missing inspector fragment: ${needle}`);
            }
            const stepThumbs = (traceHtml.match(/step-thumb/g) || []).length;
            if (stepThumbs < 2) throw new Error("purpose block timeline did not render trace-indexed steps");
            const before = run("state.selectedTraceKey + '|' + state.selectedStepIndex");
            run("render();");
            const after = run("state.selectedTraceKey + '|' + state.selectedStepIndex");
            if (before !== after) throw new Error("refresh render did not preserve selected trace/step");
            """
        ),
        encoding="utf-8",
    )

    subprocess.run(["node", str(harness), str(app_path), str(snapshot_path)], check=True)
