import json

import numpy as np
import pytest

from p2a.core import BonusMapStore


class FakeBatch:
    def __init__(self, advantages, returns, response_mask, non_tensor_batch):
        self.batch = {
            "advantages": advantages,
            "returns": returns,
            "response_mask": response_mask,
        }
        self.non_tensor_batch = non_tensor_batch


def _batch(torch_module):
    return FakeBatch(
        advantages=torch_module.ones((1, 4), dtype=torch_module.float32),
        returns=torch_module.ones((1, 4), dtype=torch_module.float32),
        response_mask=torch_module.ones((1, 4), dtype=torch_module.int64),
        non_tensor_batch={
            "instance_id": np.array(["demo__block"], dtype=object),
            "p2a_step_traces": np.array(
                [
                    [
                        {
                            "response_start": 0,
                            "response_end": 2,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": {
                                            "command": "view",
                                            "path": "/testbed/pkg/demo.py",
                                            "view_range": [50, 60],
                                        },
                                    }
                                }
                            ],
                        },
                        {
                            "response_start": 2,
                            "response_end": 4,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "str_replace_editor",
                                        "arguments": {
                                            "command": "view",
                                            "path": "/testbed/pkg/demo.py",
                                            "view_range": [1, 5],
                                        },
                                    }
                                }
                            ],
                        },
                    ]
                ],
                dtype=object,
            ),
        },
    )


def test_block_credit_reshapes_full_read_block(tmp_path):
    torch_module = pytest.importorskip("torch")
    pytest.importorskip("ray")
    from p2a.trainer import apply_p2a_reshape

    bonus_dir = tmp_path / "bonus_maps"
    bonus_dir.mkdir()
    (bonus_dir / "demo__block.json").write_text(
        json.dumps(
            {
                "instance_id": "demo__block",
                "call_graph_nodes": {
                    "pkg/demo.py::target": {
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
    store = BonusMapStore(str(bonus_dir))

    step_batch, step_metrics = apply_p2a_reshape(_batch(torch_module), store, m_max=3.0, credit_granularity="step")
    block_batch, block_metrics = apply_p2a_reshape(_batch(torch_module), store, m_max=3.0, credit_granularity="block")

    assert step_batch.batch["advantages"].tolist() == [[1.0, 1.0, 3.0, 3.0]]
    assert step_metrics["p2a/n_reshaped"] == 1
    assert block_batch.batch["advantages"].tolist() == [[3.0, 3.0, 3.0, 3.0]]
    assert block_metrics["p2a/n_reshaped"] == 1
    assert block_metrics["p2a/block_n_blocks"] == 1
    assert block_metrics["p2a/block_n_achieving"] == 1
    assert block_metrics["p2a/block_reshaped_token_share"] == 1.0
