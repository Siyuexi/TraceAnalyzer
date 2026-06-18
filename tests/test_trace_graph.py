from p2a.trace import build_call_graph_from_traces, find_modified_callables_from_sources


def test_call_graph_persists_edges_and_node_source():
    traces = [
        [
            {"file_path": "tests/test_demo.py", "line_no": 3, "func_name": "test_demo", "qualified_name": "test_demo"},
            {"file_path": "pkg/mid.py", "line_no": 2, "func_name": "mid", "qualified_name": "mid"},
            {"file_path": "pkg/demo.py", "line_no": 2, "func_name": "target", "qualified_name": "target", "is_patched": True},
        ]
    ]
    modified = [
        {
            "file_path": "pkg/demo.py",
            "qualified_name": "target",
            "name": "target",
            "start_line": 1,
            "end_line": 2,
            "source": "def target():\n    return 1",
        },
        {
            "file_path": "pkg/unused.py",
            "qualified_name": "unused",
            "name": "unused",
            "start_line": 1,
            "end_line": 2,
            "source": "def unused():\n    return 1",
        }
    ]
    sources = {
        "tests/test_demo.py": "def test_demo():\n    mid()\n",
        "pkg/mid.py": "def mid():\n    target()\n",
        "pkg/demo.py": "def target():\n    return 1\n",
    }

    result = build_call_graph_from_traces(traces, modified, file_reader=sources.get)

    assert result["call_graph_edges"] == [
        ["pkg/mid.py::mid", "pkg/demo.py::target"],
        ["tests/test_demo.py::test_demo", "pkg/mid.py::mid"],
    ]
    assert result["call_graph_nodes"]["pkg/demo.py::target"]["source"] == "def target():\n    return 1"
    assert result["call_graph_nodes"]["pkg/mid.py::mid"]["source"] == "def mid():\n    target()"
    assert "source" not in result["patched_callables"][0]
    assert "source" not in result["unobserved_patched_callables"][0]


def test_modified_callable_metadata_carries_source_for_node_enrichment():
    modified = find_modified_callables_from_sources(
        "def target():\n    return 1\n",
        "def target():\n    return 2\n",
        "pkg/demo.py",
    )

    assert modified[0]["source"] == "def target():\n    return 1"
