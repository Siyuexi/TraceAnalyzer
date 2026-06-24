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
            "by_case_type": {},
            "trends": [],
        },
        "experiments": [
            {
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
        "model_metrics": [],
        "runs": [],
        "details": [
            {
                "experiment_key": "third_party_api::exp-a::internal_api::swebench-hard::model-a::model-a",
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
                        "thought": "inspect symptom",
                        "response_text": "view symptom",
                        "tool_calls": [{"function": {"name": "str_replace_editor"}}],
                        "tool_results": [{"observation": "symptom body"}],
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
                        "thought": "inspect root",
                        "response_text": "view root",
                        "tool_calls": [{"function": {"name": "str_replace_editor"}}],
                        "tool_results": [{"observation": "root body"}],
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
            if (run("state.selectedExperimentKey") !== null) {
              throw new Error("multi-experiment overview must require explicit selection");
            }
            const expHtml = elements.get("experiment-table").innerHTML;
            if (!expHtml.includes("Inspect") || !expHtml.includes("exp-a") || !expHtml.includes("exp-b")) {
              throw new Error("experiment registry did not render both experiments");
            }
            run(`state.selectedExperimentKey = ${JSON.stringify(snapshot.experiments[0].experiment_key)};
                 state.selectedTraceKey = null;
                 state.selectedStepIndex = 1;
                 render();`);
            const traceHtml = elements.get("trace-inspector").innerHTML;
            for (const needle of ["trace-left", "trace-middle", "trace-right", "<svg", "Purpose blocks", "Tool calls", "Matched bonus-map nodes"]) {
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
