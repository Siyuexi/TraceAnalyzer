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
    assert result["raw_hop_max"] == 2
    assert result["hop_max"] == 1
    test_node = result["call_graph_nodes"]["tests/test_demo.py::test_demo"]
    assert test_node["rewardable"] is False
    assert test_node["node_role"] == "test_harness"
    assert test_node["excluded_from_hop_max"] is True
    assert test_node["normalized_distance"] == 1.0
    assert result["test_harness_file_patterns"] == ["tests/**"]
    assert result["excluded_test_harness_nodes"] == ["tests/test_demo.py::test_demo"]


def test_call_graph_uses_nested_helper_source_for_nested_frames():
    traces = [
        [
            {"file_path": "tests/test_nested.py", "line_no": 5, "func_name": "C.outer", "qualified_name": "C.outer"},
            {"file_path": "tests/test_nested.py", "line_no": 4, "func_name": "C.inner", "qualified_name": "C.inner"},
            {"file_path": "pkg/target.py", "line_no": 2, "func_name": "target", "qualified_name": "target", "is_patched": True},
        ]
    ]
    modified = [
        {
            "file_path": "pkg/target.py",
            "qualified_name": "target",
            "name": "target",
            "start_line": 1,
            "end_line": 2,
            "source": "def target():\n    return 1",
        },
    ]
    sources = {
        "tests/test_nested.py": (
            "class C:\n"
            "    def outer(self):\n"
            "        def inner():\n"
            "            target()\n"
            "        inner()\n"
        ),
        "pkg/target.py": "def target():\n    return 1\n",
    }

    result = build_call_graph_from_traces(traces, modified, file_reader=sources.get)

    outer = result["call_graph_nodes"]["tests/test_nested.py::C.outer"]
    inner = result["call_graph_nodes"]["tests/test_nested.py::C.inner"]
    assert (outer["start_line"], outer["end_line"]) == (2, 5)
    assert outer["source"] == "    def outer(self):\n        def inner():\n            target()\n        inner()"
    assert (inner["start_line"], inner["end_line"]) == (3, 4)
    assert inner["source"] == "        def inner():\n            target()"


def test_modified_callable_metadata_carries_source_for_node_enrichment():
    modified = find_modified_callables_from_sources(
        "def target():\n    return 1\n",
        "def target():\n    return 2\n",
        "pkg/demo.py",
    )

    assert modified[0]["source"] == "def target():\n    return 1"


def test_shallow_src_callers_remain_rewardable():
    traces = [
        [
            {"file_path": "tests/test_x.py", "line_no": 1, "func_name": "test_x", "qualified_name": "test_x"},
            {"file_path": "src/helper.py", "line_no": 2, "func_name": "helper", "qualified_name": "helper"},
            {"file_path": "src/target.py", "line_no": 3, "func_name": "target", "qualified_name": "target", "is_patched": True},
        ]
    ]
    modified = [
        {
            "file_path": "src/target.py",
            "qualified_name": "target",
            "name": "target",
            "start_line": 3,
            "end_line": 4,
            "source": "def target():\n    return 1",
        }
    ]

    result = build_call_graph_from_traces(traces, modified)

    helper = result["call_graph_nodes"]["src/helper.py::helper"]
    test_node = result["call_graph_nodes"]["tests/test_x.py::test_x"]
    assert result["raw_hop_max"] == 2
    assert result["hop_max"] == 1
    assert helper["rewardable"] is True
    assert helper["node_role"] == "program"
    assert helper["hop_distance"] == 1
    assert helper["normalized_distance"] == 1.0
    assert test_node["rewardable"] is False
    assert test_node["node_role"] == "test_harness"


def test_tests_py_modules_are_test_harnesses_even_inside_reward_prefix():
    traces = [
        [
            {"file_path": "myapp/tests.py", "line_no": 1, "func_name": "test_model", "qualified_name": "test_model"},
            {"file_path": "myapp/utils.py", "line_no": 2, "func_name": "helper", "qualified_name": "helper"},
            {"file_path": "myapp/models.py", "line_no": 3, "func_name": "Model.save", "qualified_name": "Model.save", "is_patched": True},
        ]
    ]
    modified = [
        {
            "file_path": "myapp/models.py",
            "qualified_name": "Model.save",
            "name": "save",
            "start_line": 3,
            "end_line": 4,
            "source": "def save(self):\n    return None",
        }
    ]

    result = build_call_graph_from_traces(traces, modified)

    test_node = result["call_graph_nodes"]["myapp/tests.py::test_model"]
    helper = result["call_graph_nodes"]["myapp/utils.py::helper"]
    assert result["raw_hop_max"] == 2
    assert result["hop_max"] == 1
    assert result["test_harness_file_patterns"] == ["tests.py"]
    assert result["excluded_test_harness_nodes"] == ["myapp/tests.py::test_model"]
    assert test_node["rewardable"] is False
    assert test_node["node_role"] == "test_harness"
    assert test_node["normalized_distance"] == 1.0
    assert helper["rewardable"] is True
    assert helper["node_role"] == "program"
    assert helper["hop_distance"] == 1


def test_django_test_client_prefix_does_not_determine_hop_max():
    traces = [
        [
            {
                "file_path": "tests/admin_inlines/tests.py",
                "line_no": 10,
                "func_name": "TestInlinePermissions.test_inline_change_m2m_view_only_perm",
                "qualified_name": "TestInlinePermissions.test_inline_change_m2m_view_only_perm",
            },
            {
                "file_path": "django/test/client.py",
                "line_no": 20,
                "func_name": "Client.get",
                "qualified_name": "Client.get",
            },
            {
                "file_path": "django/core/handlers/base.py",
                "line_no": 30,
                "func_name": "ClientHandler.get_response",
                "qualified_name": "ClientHandler.get_response",
            },
            {
                "file_path": "django/utils/deprecation.py",
                "line_no": 40,
                "func_name": "SessionMiddleware.__call__",
                "qualified_name": "SessionMiddleware.__call__",
            },
            {
                "file_path": "django/contrib/admin/options.py",
                "line_no": 50,
                "func_name": "AuthorAdmin.wrapper",
                "qualified_name": "AuthorAdmin.wrapper",
            },
            {
                "file_path": "django/utils/decorators.py",
                "line_no": 60,
                "func_name": "_wrapped_view",
                "qualified_name": "_wrapped_view",
            },
            {
                "file_path": "django/contrib/admin/options.py",
                "line_no": 70,
                "func_name": "BookInline.has_change_permission",
                "qualified_name": "BookInline.has_change_permission",
                "is_patched": True,
            },
        ]
    ]
    modified = [
        {
            "file_path": "django/contrib/admin/options.py",
            "qualified_name": "BookInline.has_change_permission",
            "name": "has_change_permission",
            "start_line": 70,
            "end_line": 72,
            "source": "def has_change_permission(self, request, obj=None):\n    return True",
        }
    ]

    result = build_call_graph_from_traces(traces, modified)

    assert result["raw_hop_max"] == 6
    assert result["hop_max"] == 2
    assert result["rewardable_node_count"] == 3
    assert result["excluded_test_harness_node_count"] == 2
    assert result["excluded_symptom_prefix_node_count"] == 2
    assert result["test_harness_file_patterns"] == ["django/test/**", "tests/**"]

    test_node = result["call_graph_nodes"]["django/test/client.py::Client.get"]
    assert test_node["rewardable"] is False
    assert test_node["node_role"] == "test_harness"
    assert test_node["normalized_distance"] == 1.0

    symptom_node = result["call_graph_nodes"]["django/core/handlers/base.py::ClientHandler.get_response"]
    assert symptom_node["rewardable"] is False
    assert symptom_node["node_role"] == "symptom_prefix"
    assert symptom_node["exclusion_reason"] == "symptom_prefix"
    assert symptom_node["normalized_distance"] == 1.0

    wrapper = result["call_graph_nodes"]["django/contrib/admin/options.py::AuthorAdmin.wrapper"]
    assert wrapper["rewardable"] is True
    assert wrapper["hop_distance"] == 2
    assert wrapper["normalized_distance"] == 1.0

    patched = result["call_graph_nodes"]["django/contrib/admin/options.py::BookInline.has_change_permission"]
    assert patched["rewardable"] is True
    assert patched["hop_distance"] == 0
    assert patched["normalized_distance"] == 0.0
