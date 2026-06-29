import json

from p2a.bonus_map_scope import (
    EXPOSED_CASE,
    LATENT_CASE,
    bonus_map_pattern_computable,
    canonical_bonus_case_type,
    enrich_bonus_map_case_metadata,
    parse_bonus_map_instance_filter,
    select_rows_by_bonus_map_scope,
)


def _bonus_map(*, instance_id: str, anchors: list[str], roots: list[str], edges: list[list[str]]):
    nodes = {
        "pkg/symptom.py::symptom": {"normalized_distance": 1.0},
        "pkg/mid.py::mid": {"normalized_distance": 0.5},
        "pkg/root.py::root": {"normalized_distance": 0.0},
    }
    return {
        "instance_id": instance_id,
        "case_type": "standard",
        "traceable": True,
        "selected_issue_anchor_nodes": anchors,
        "root_cause_nodes": roots,
        "reward_path_edges": edges,
        "call_graph_nodes": nodes,
    }


def test_legacy_standard_classifies_as_latent_when_clean_pattern_path_exists():
    bonus_map = _bonus_map(
        instance_id="case-1",
        anchors=["pkg/symptom.py::symptom"],
        roots=["pkg/root.py::root"],
        edges=[
            ["pkg/symptom.py::symptom", "pkg/mid.py::mid"],
            ["pkg/mid.py::mid", "pkg/root.py::root"],
        ],
    )

    assert canonical_bonus_case_type(bonus_map) == LATENT_CASE
    assert bonus_map_pattern_computable(bonus_map) is True

    enriched = enrich_bonus_map_case_metadata(bonus_map)

    assert enriched["case_type"] == LATENT_CASE
    assert enriched["legacy_case_type"] == "standard"
    assert enriched["root_anchor_overlap"] == "none"
    assert enriched["pattern_computable"] is True


def test_legacy_standard_classifies_as_exposed_when_all_roots_are_anchors():
    bonus_map = _bonus_map(
        instance_id="case-1",
        anchors=["pkg/root.py::root"],
        roots=["pkg/root.py::root"],
        edges=[["pkg/root.py::root", "pkg/mid.py::mid"]],
    )

    assert canonical_bonus_case_type(bonus_map) == EXPOSED_CASE
    assert bonus_map_pattern_computable(bonus_map) is False


def test_legacy_standard_classifies_partial_overlap_as_exposed():
    bonus_map = _bonus_map(
        instance_id="case-1",
        anchors=["pkg/symptom.py::symptom", "pkg/root.py::root"],
        roots=["pkg/root.py::root", "pkg/other.py::root"],
        edges=[["pkg/symptom.py::symptom", "pkg/root.py::root"]],
    )

    assert canonical_bonus_case_type(bonus_map) == EXPOSED_CASE
    assert bonus_map_pattern_computable(bonus_map) is False


def test_legacy_standard_without_path_edges_is_exposed():
    bonus_map = _bonus_map(
        instance_id="case-1",
        anchors=["pkg/symptom.py::symptom"],
        roots=["pkg/root.py::root"],
        edges=[],
    )

    assert canonical_bonus_case_type(bonus_map) == EXPOSED_CASE
    assert bonus_map_pattern_computable(bonus_map) is False


def test_raw_latent_without_clean_pattern_structure_is_exposed():
    bonus_map = _bonus_map(
        instance_id="case-1",
        anchors=["pkg/symptom.py::symptom", "pkg/root.py::root"],
        roots=["pkg/root.py::root"],
        edges=[["pkg/symptom.py::symptom", "pkg/root.py::root"]],
    )
    bonus_map["case_type"] = LATENT_CASE

    assert canonical_bonus_case_type(bonus_map) == EXPOSED_CASE
    assert bonus_map_pattern_computable(bonus_map) is False


def test_scope_filter_selects_rows_by_bonus_map_case_type(tmp_path):
    bonus_dir = tmp_path / "bonus"
    bonus_dir.mkdir()
    (bonus_dir / "case-1.json").write_text(
        json.dumps(
            _bonus_map(
                instance_id="case-1",
                anchors=["pkg/symptom.py::symptom"],
                roots=["pkg/root.py::root"],
                edges=[
                    ["pkg/symptom.py::symptom", "pkg/mid.py::mid"],
                    ["pkg/mid.py::mid", "pkg/root.py::root"],
                ],
            )
        ),
        encoding="utf-8",
    )
    (bonus_dir / "case-2.json").write_text(
        json.dumps(
            _bonus_map(
                instance_id="case-2",
                anchors=["pkg/root.py::root"],
                roots=["pkg/root.py::root"],
                edges=[],
            )
        ),
        encoding="utf-8",
    )

    scoped = select_rows_by_bonus_map_scope(
        [{"instance_id": "case-1"}, {"instance_id": "case-2"}],
        bonus_map_dir=bonus_dir,
        instance_id=lambda row: row["instance_id"],
        scope_filter=parse_bonus_map_instance_filter({"case_type": "latent", "pattern_computable": True}),
    )

    assert scoped.rows == [{"instance_id": "case-1"}]
    assert scoped.metadata["source_size"] == 2
    assert scoped.metadata["selected_size"] == 1
    assert scoped.metadata["matched_case_types"] == {"latent": 1}
