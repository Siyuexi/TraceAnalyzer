from p2a.trace import build_call_graph_from_traces, find_modified_callables_from_sources


def _frame(file_path: str, qualified_name: str, line_no: int, *, patched: bool = False) -> dict:
    frame = {
        "file_path": file_path,
        "line_no": line_no,
        "func_name": qualified_name,
        "qualified_name": qualified_name,
    }
    if patched:
        frame["is_patched"] = True
    return frame


def _modified(file_path: str, qualified_name: str, start_line: int = 1, end_line: int = 2) -> dict:
    return {
        "file_path": file_path,
        "qualified_name": qualified_name,
        "name": qualified_name.rsplit(".", 1)[-1],
        "start_line": start_line,
        "end_line": end_line,
        "source": f"def {qualified_name.rsplit('.', 1)[-1]}():\n    pass",
    }


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
    assert helper["node_role"] == "intermediate"
    assert helper["hop_distance"] == 1
    assert helper["normalized_distance"] == 1.0
    assert test_node["rewardable"] is False
    assert test_node["node_role"] == "test_harness"
    assert result["reward_start_source"] == "test_filtered_fallback"


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
    assert helper["node_role"] == "intermediate"
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
    assert result["hop_max"] == 4
    assert result["rewardable_node_count"] == 5
    assert result["excluded_test_harness_node_count"] == 2
    assert result["excluded_pre_symptom_node_count"] == 0
    assert result["test_harness_file_patterns"] == ["django/test/**", "tests/**"]
    assert result["reward_start_source"] == "test_filtered_fallback"

    test_node = result["call_graph_nodes"]["django/test/client.py::Client.get"]
    assert test_node["rewardable"] is False
    assert test_node["node_role"] == "test_harness"
    assert test_node["normalized_distance"] == 1.0

    framework_node = result["call_graph_nodes"]["django/core/handlers/base.py::ClientHandler.get_response"]
    assert framework_node["rewardable"] is True
    assert framework_node["node_role"] == "intermediate"
    assert framework_node["hop_distance"] == 4
    assert framework_node["normalized_distance"] == 1.0

    wrapper = result["call_graph_nodes"]["django/contrib/admin/options.py::AuthorAdmin.wrapper"]
    assert wrapper["rewardable"] is True
    assert wrapper["hop_distance"] == 2
    assert wrapper["normalized_distance"] == 0.5

    patched = result["call_graph_nodes"]["django/contrib/admin/options.py::BookInline.has_change_permission"]
    assert patched["rewardable"] is True
    assert patched["node_role"] == "root_cause"
    assert patched["hop_distance"] == 0
    assert patched["normalized_distance"] == 0.0


def test_issue_anchor_marks_pre_symptom_frames_non_rewardable():
    traces = [
        [
            _frame("tests/test_issue.py", "test_issue", 1),
            _frame("framework/request.py", "dispatch", 10),
            _frame("app/views.py", "symptom", 20),
            _frame("app/service.py", "intermediate", 30),
            _frame("app/root.py", "patched_root", 40, patched=True),
        ]
    ]
    modified = [_modified("app/root.py", "patched_root", 40, 42)]

    result = build_call_graph_from_traces(
        traces,
        modified,
        issue_text="The user-visible failure starts in `symptom`.",
    )

    framework = result["call_graph_nodes"]["framework/request.py::dispatch"]
    symptom = result["call_graph_nodes"]["app/views.py::symptom"]
    intermediate = result["call_graph_nodes"]["app/service.py::intermediate"]
    root = result["call_graph_nodes"]["app/root.py::patched_root"]

    assert result["reward_start_source"] == "issue_anchor"
    assert result["selected_issue_anchor_nodes"] == ["app/views.py::symptom"]
    assert result["excluded_pre_symptom_nodes"] == ["framework/request.py::dispatch"]
    assert framework["rewardable"] is False
    assert framework["node_role"] == "pre_symptom"
    assert framework["excluded_from_hop_max"] is True
    assert symptom["rewardable"] is True
    assert symptom["node_role"] == "symptom"
    assert intermediate["node_role"] == "intermediate"
    assert root["node_role"] == "root_cause"
    assert result["hop_max"] == 2


def test_issue_anchor_uses_deepest_matching_symptom_candidate():
    traces = [
        [
            _frame("tests/test_nested.py", "test_nested", 1),
            _frame("pkg/outer.py", "outer", 10),
            _frame("pkg/inner.py", "inner_symptom", 20),
            _frame("pkg/root.py", "patched_root", 30, patched=True),
        ]
    ]
    modified = [_modified("pkg/root.py", "patched_root", 30, 32)]

    result = build_call_graph_from_traces(
        traces,
        modified,
        issue_text="Both `outer` and `inner_symptom` appear in the report.",
    )

    outer = result["call_graph_nodes"]["pkg/outer.py::outer"]
    inner = result["call_graph_nodes"]["pkg/inner.py::inner_symptom"]
    assert result["selected_issue_anchor_nodes"] == ["pkg/inner.py::inner_symptom"]
    assert outer["rewardable"] is False
    assert outer["node_role"] == "pre_symptom"
    assert inner["rewardable"] is True
    assert inner["node_role"] == "symptom"


def test_generic_issue_anchor_names_do_not_anchor_without_context():
    traces = [
        [
            _frame("tests/test_generic.py", "test_generic", 1),
            _frame("pkg/helpers.py", "inner", 10),
            _frame("pkg/wrappers.py", "wrapper", 20),
            _frame("pkg/views.py", "View.get", 30),
            _frame("pkg/root.py", "patched_root", 40, patched=True),
        ]
    ]
    modified = [_modified("pkg/root.py", "patched_root", 40, 42)]

    result = build_call_graph_from_traces(
        traces,
        modified,
        issue_text="The issue mentions `inner`, `wrapper`, get(), and `__call__` generically.",
    )

    assert result["reward_start_source"] == "test_filtered_fallback"
    assert result["selected_issue_anchor_nodes"] == []
    assert result["excluded_pre_symptom_nodes"] == []
    for key, node in result["call_graph_nodes"].items():
        if key.startswith("pkg/"):
            assert node["rewardable"] is True


def test_disambiguated_generic_issue_anchor_can_match():
    traces = [
        [
            _frame("tests/test_context.py", "test_context", 1),
            _frame("framework/router.py", "dispatch", 10),
            _frame("pkg/views.py", "View.get", 20),
            _frame("pkg/root.py", "patched_root", 30, patched=True),
        ]
    ]
    modified = [_modified("pkg/root.py", "patched_root", 30, 32)]

    result = build_call_graph_from_traces(
        traces,
        modified,
        issue_text="Traceback points at `View.get`.",
    )

    assert result["reward_start_source"] == "issue_anchor"
    assert result["selected_issue_anchor_nodes"] == ["pkg/views.py::View.get"]
    assert result["call_graph_nodes"]["framework/router.py::dispatch"]["node_role"] == "pre_symptom"
    assert result["call_graph_nodes"]["pkg/views.py::View.get"]["node_role"] == "symptom"


def test_upstream_patched_callable_gets_positive_distance_from_deeper_root():
    traces = [
        [
            _frame("tests/test_root.py", "test_root", 1),
            _frame("pkg/foo.py", "foo1", 10, patched=True),
            _frame("pkg/mid.py", "mid", 20),
            _frame("pkg/foo.py", "foo2", 30, patched=True),
        ]
    ]
    modified = [
        _modified("pkg/foo.py", "foo1", 10, 12),
        _modified("pkg/foo.py", "foo2", 30, 32),
    ]

    result = build_call_graph_from_traces(traces, modified)

    foo1 = result["call_graph_nodes"]["pkg/foo.py::foo1"]
    foo2 = result["call_graph_nodes"]["pkg/foo.py::foo2"]
    assert foo2["hop_distance"] == 0
    assert foo2["normalized_distance"] == 0.0
    assert foo1["rewardable"] is True
    assert foo1["hop_distance"] > 0
    assert result["patched_root_selection"]["terminal_root_seeds"] == ["pkg/foo.py::foo2"]
    assert result["patched_root_selection"]["upstream_adapter_patched_callables"] == ["pkg/foo.py::foo1"]
    assert result["patched_root_selection"]["legacy_distance_zero_node_count"] == 2
    assert result["patched_root_selection"]["distance_zero_node_count"] == 1


def test_independent_patched_callables_remain_separate_roots():
    traces = [
        [
            _frame("tests/test_a.py", "test_a", 1),
            _frame("pkg/a.py", "foo1", 10, patched=True),
        ],
        [
            _frame("tests/test_b.py", "test_b", 1),
            _frame("pkg/b.py", "foo2", 20, patched=True),
        ],
    ]
    modified = [
        _modified("pkg/a.py", "foo1", 10, 12),
        _modified("pkg/b.py", "foo2", 20, 22),
    ]

    result = build_call_graph_from_traces(traces, modified)

    assert result["call_graph_nodes"]["pkg/a.py::foo1"]["normalized_distance"] == 0.0
    assert result["call_graph_nodes"]["pkg/b.py::foo2"]["normalized_distance"] == 0.0
    assert result["patched_root_selection"]["terminal_root_seeds"] == ["pkg/a.py::foo1", "pkg/b.py::foo2"]
    assert result["patched_root_selection"]["patched_dependency_edges"] == []


def test_mixed_role_patched_callable_is_not_global_root():
    traces = [
        [
            _frame("tests/test_short.py", "test_short", 1),
            _frame("pkg/foo.py", "foo1", 10, patched=True),
        ],
        [
            _frame("tests/test_long.py", "test_long", 1),
            _frame("pkg/foo.py", "foo1", 10, patched=True),
            _frame("pkg/mid.py", "mid", 20),
            _frame("pkg/foo.py", "foo2", 30, patched=True),
        ],
    ]
    modified = [
        _modified("pkg/foo.py", "foo1", 10, 12),
        _modified("pkg/foo.py", "foo2", 30, 32),
    ]

    result = build_call_graph_from_traces(traces, modified)

    assert result["call_graph_nodes"]["pkg/foo.py::foo2"]["normalized_distance"] == 0.0
    assert result["call_graph_nodes"]["pkg/foo.py::foo1"]["hop_distance"] > 0
    trace_frames = result["patched_root_selection"]["observed_patched_frames_by_trace"]
    short_foo1 = trace_frames[0]["patched_frames"][0]
    long_foo1 = trace_frames[1]["patched_frames"][0]
    assert short_foo1["trace_terminal"] is True
    assert short_foo1["selected_root_seed"] is False
    assert short_foo1["upstream_adapter"] is True
    assert long_foo1["trace_terminal"] is False
    assert long_foo1["downstream_patched_frame_keys"] == ["pkg/foo.py::foo2"]


def test_cyclic_patched_dependency_component_remains_traceable():
    traces = [
        [
            _frame("tests/test_ab.py", "test_ab", 1),
            _frame("pkg/a.py", "A", 10, patched=True),
            _frame("pkg/b.py", "B", 20, patched=True),
        ],
        [
            _frame("tests/test_ba.py", "test_ba", 1),
            _frame("pkg/b.py", "B", 20, patched=True),
            _frame("pkg/a.py", "A", 10, patched=True),
        ],
    ]
    modified = [
        _modified("pkg/a.py", "A", 10, 12),
        _modified("pkg/b.py", "B", 20, 22),
    ]

    result = build_call_graph_from_traces(traces, modified)

    assert result["traceable"] is True
    assert result["call_graph_nodes"]["pkg/a.py::A"]["normalized_distance"] == 0.0
    assert result["call_graph_nodes"]["pkg/b.py::B"]["normalized_distance"] == 0.0
    assert result["patched_root_selection"]["terminal_root_seeds"] == ["pkg/a.py::A", "pkg/b.py::B"]
    assert result["patched_root_selection"]["terminal_root_components"] == [["pkg/a.py::A", "pkg/b.py::B"]]
    assert result["patched_root_selection"]["patched_dependency_edges"] == [
        ["pkg/a.py::A", "pkg/b.py::B"],
        ["pkg/b.py::B", "pkg/a.py::A"],
    ]


def test_overlapping_traces_use_deepest_terminal_patched_root():
    traces = [
        [
            _frame("tests/test_long.py", "test_long", 1),
            _frame("pkg/foo.py", "foo1", 10),
            _frame("pkg/foo.py", "foo2", 20, patched=True),
            _frame("pkg/foo.py", "foo3", 30, patched=True),
        ],
        [
            _frame("tests/test_short.py", "test_short", 1),
            _frame("pkg/foo.py", "foo2", 20, patched=True),
            _frame("pkg/foo.py", "foo3", 30, patched=True),
        ],
    ]
    modified = [
        _modified("pkg/foo.py", "foo2", 20, 22),
        _modified("pkg/foo.py", "foo3", 30, 32),
    ]

    result = build_call_graph_from_traces(traces, modified)

    foo1 = result["call_graph_nodes"]["pkg/foo.py::foo1"]
    foo2 = result["call_graph_nodes"]["pkg/foo.py::foo2"]
    foo3 = result["call_graph_nodes"]["pkg/foo.py::foo3"]
    assert result["patched_root_selection"]["terminal_root_seeds"] == ["pkg/foo.py::foo3"]
    assert foo3["normalized_distance"] == 0.0
    assert foo2["hop_distance"] == 1
    assert foo1["hop_distance"] == 2
    assert result["hop_max"] == 2


def test_hop_max_uses_rewardable_program_nodes_with_shared_root():
    traces = [
        [
            _frame("tests/test_long.py", "test_long", 1),
            _frame("pkg/a.py", "a", 10),
            _frame("pkg/b.py", "b", 20),
            _frame("pkg/root.py", "root", 30, patched=True),
        ],
        [
            _frame("tests/test_short.py", "test_short", 1),
            _frame("pkg/root.py", "root", 30, patched=True),
        ],
    ]
    modified = [_modified("pkg/root.py", "root", 30, 32)]

    result = build_call_graph_from_traces(traces, modified)

    assert result["hop_max"] == 2
    assert result["call_graph_nodes"]["pkg/a.py::a"]["normalized_distance"] == 1.0
    assert result["call_graph_nodes"]["pkg/b.py::b"]["normalized_distance"] == 0.5
    for key, node in result["call_graph_nodes"].items():
        if key.startswith("tests/"):
            assert node["rewardable"] is False
            assert node["normalized_distance"] == 1.0


def test_orange3_70a4df3348_regression_uses_number_of_decimals_as_root():
    traces = [
        [
            _frame("r2e_tests/test_1.py", "TestContinuousVariable.test_decimals", 8),
            _frame("Orange/data/variable.py", "ContinuousVariable.__init__", 525, patched=True),
            _frame("Orange/data/variable.py", "ContinuousVariable.number_of_decimals", 560, patched=True),
        ]
    ]
    modified = [
        _modified("Orange/data/variable.py", "ContinuousVariable.__init__", 519, 530),
        _modified("Orange/data/variable.py", "ContinuousVariable.number_of_decimals", 559, 562),
    ]

    result = build_call_graph_from_traces(traces, modified)

    init = result["call_graph_nodes"]["Orange/data/variable.py::ContinuousVariable.__init__"]
    number = result["call_graph_nodes"]["Orange/data/variable.py::ContinuousVariable.number_of_decimals"]
    assert number["hop_distance"] == 0
    assert number["normalized_distance"] == 0.0
    assert init["rewardable"] is True
    assert init["hop_distance"] == 1
    assert init["normalized_distance"] == 1.0
    assert result["patched_root_selection"]["terminal_root_seeds"] == [
        "Orange/data/variable.py::ContinuousVariable.number_of_decimals"
    ]
    assert result["patched_root_selection"]["upstream_adapter_patched_callables"] == [
        "Orange/data/variable.py::ContinuousVariable.__init__"
    ]
