import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location("swebench_pro_images", ROOT / "scripts" / "swebench_pro_images.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_records_resolve_source_and_mirror_refs(tmp_path):
    module = _load_module()
    path = tmp_path / "swe_bench_pro.parquet"
    pd.DataFrame(
        [
            {
                "instance_id": "demo__1",
                "repo": "qutebrowser/qutebrowser",
                "repo_language": "python",
                "dockerhub_tag": "qutebrowser.demo",
                "extra_info": {"tools_kwargs": {"reward": {"metadata": {"data_source": "swebench-pro"}}}},
            },
            {
                "instance_id": "demo__2",
                "repo": "elastic/kibana",
                "repo_language": "ts",
                "dockerhub_tag": "kibana.demo",
                "extra_info": {"tools_kwargs": {"reward": {"metadata": {"data_source": "swebench-pro"}}}},
            },
        ]
    ).to_parquet(path, index=False)

    records = module.image_records(path)

    assert len(records) == 1
    assert records[0]["source_image"] == "jefzda/sweap-images:qutebrowser.demo"
    assert records[0]["mirror_image"] == "pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images:qutebrowser.demo"


def test_manifest_status_reports_missing(monkeypatch):
    module = _load_module()

    class Proc:
        returncode = 1
        stderr = "unknown: repository code/sweap-images not found\n"

    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: Proc())

    assert module.manifest_status("pair-diag.example/code/sweap-images:demo", timeout=1) == {
        "status": "missing",
        "error": "unknown: repository code/sweap-images not found",
    }


def test_mirror_commands_emit_pull_tag_push_once():
    module = _load_module()

    commands = module.mirror_commands(
        [
            {
                "instance_id": "demo__1",
                "source_image": "jefzda/sweap-images:demo",
                "mirror_image": "pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images:demo",
            },
            {
                "instance_id": "demo__1-duplicate",
                "source_image": "jefzda/sweap-images:demo",
                "mirror_image": "pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images:demo",
            },
        ]
    )

    text = "\n".join(commands)
    assert text.count("docker pull jefzda/sweap-images:demo") == 1
    assert "docker tag jefzda/sweap-images:demo pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images:demo" in text
    assert text.count("docker push pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images:demo") == 1
