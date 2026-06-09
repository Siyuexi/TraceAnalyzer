import json

import numpy as np

from p2a.validation_metrics import compute_validation_p2a_metrics, validation_records_from_batch


class FakeBatch:
    def __init__(self, non_tensor_batch):
        self.non_tensor_batch = non_tensor_batch


def test_validation_records_from_extra_fields_and_metric_flattening(tmp_path):
    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__abc123.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__abc123",
                "case_type": "standard",
                "traceable": True,
                "call_graph_nodes": {
                    "pkg.demo:demo": {
                        "file_path": "pkg/demo.py",
                        "start_line": 1,
                        "end_line": 5,
                        "normalized_distance": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    batch = FakeBatch(
        {
            "uid": np.array(["uid-1"], dtype=object),
            "data_source": np.array(["swebench"], dtype=object),
            "extra_fields": np.array(
                [
                    {
                        "instance_id": "demo__abc123",
                        "p2a_step_traces": [
                            {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "execute_bash",
                                            "arguments": {"command": "cat /testbed/pkg/demo.py"},
                                        }
                                    }
                                ]
                            }
                        ],
                    }
                ],
                dtype=object,
            ),
        }
    )

    records = validation_records_from_batch(batch, output_texts=[""], scores=[1.0])
    assert records[0]["instance_id"] == "demo__abc123"
    assert records[0]["data_source"] == "swebench"

    metrics, details = compute_validation_p2a_metrics(
        records,
        bonus_map_dir=str(bonus_dir),
        tracking_mode="view_and_bash",
        near_threshold=0.5,
        m_max=3.0,
    )

    assert details[0]["has_step_traces"] is True
    assert details[0]["hit_call_graph"] is True
    assert details[0]["hit_ground_truth"] is True
    assert metrics["val-p2a/swebench/bonus_map_coverage"] == 1.0
    assert metrics["val-p2a/swebench/call_graph_coverage"] == 1.0
    assert metrics["val-p2a/swebench/read_rate"] == 1.0
    assert metrics["val-p2a/swebench/graph_hit_rate_over_call_graphs"] == 1.0
    assert metrics["val-p2a/swebench/ground_truth_hit_rate_over_call_graphs"] == 1.0
    assert metrics["val-p2a/swebench/avg_min_distance_on_hits"] == 0.0
