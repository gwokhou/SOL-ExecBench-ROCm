from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
DOWNLOAD_PATH = REPO_ROOT / "scripts" / "download_solexecbench.py"
spec = importlib.util.spec_from_file_location("download_solexecbench", DOWNLOAD_PATH)
assert spec is not None
download_solexecbench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(download_solexecbench)


def _row(name: str = "matmul_demo") -> dict:
    return {
        "name": name,
        "hf_id": "hf-demo",
        "description": "demo problem",
        "axes": json.dumps({"M": {"type": "const", "value": 2}}),
        "custom_inputs_entrypoint": None,
        "inputs": json.dumps({"x": {"shape": ["M"], "dtype": "float32"}}),
        "outputs": json.dumps({"out": {"shape": ["M"], "dtype": "float32"}}),
        "reference": "def run(x):\n    return x\n",
        "workloads": json.dumps(
            [{"uuid": "demo-workload", "axes": {}, "inputs": {"x": {"type": "random"}}}]
        ),
    }


def test_downloader_honors_category_output_root_revision_and_manifest(
    tmp_path, monkeypatch
):
    calls = []

    def fake_load_dataset(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return [_row()]

    monkeypatch.setattr(download_solexecbench, "load_dataset", fake_load_dataset)
    output_root = tmp_path / "dataset"
    manifest_path = tmp_path / "artifacts" / "manifest.json"

    rc = download_solexecbench.main(
        [
            "--category",
            "L1",
            "--output-root",
            str(output_root),
            "--manifest",
            str(manifest_path),
            "--revision",
            "abc123",
        ]
    )

    assert rc == 0
    assert calls == [
        ("nvidia/SOL-ExecBench", {"name": "L1", "split": "train", "revision": "abc123"})
    ]
    problem_dir = output_root / "L1" / "matmul_demo"
    assert (problem_dir / "definition.json").is_file()
    assert (problem_dir / "reference.py").is_file()
    assert (problem_dir / "workload.jsonl").is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_categories"] == ["L1"]
    assert manifest["source"]["revision"] == "abc123"
    assert manifest["claim_boundary"]["rocm_readiness"] is False


def test_downloader_normalizes_empty_hf_id_to_null(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_solexecbench,
        "load_dataset",
        lambda *args, **kwargs: [_row() | {"hf_id": ""}],
    )
    output_root = tmp_path / "dataset"

    rc = download_solexecbench.main(
        ["--category", "FlashInfer-Bench", "--output-root", str(output_root)]
    )

    assert rc == 0
    definition = json.loads(
        (
            output_root / "FlashInfer-Bench" / "matmul_demo" / "definition.json"
        ).read_text(encoding="utf-8")
    )
    assert definition["hf_id"] is None


def test_downloader_reuses_identical_files_and_rejects_divergent_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        download_solexecbench, "load_dataset", lambda *args, **kwargs: [_row()]
    )
    output_root = tmp_path / "dataset"
    args = ["--category", "L1", "--output-root", str(output_root)]

    assert download_solexecbench.main(args) == 0
    assert download_solexecbench.main(args) == 0

    definition_path = output_root / "L1" / "matmul_demo" / "definition.json"
    definition_path.write_text('{"local":"edit"}\n', encoding="utf-8")

    assert download_solexecbench.main(args) == 1
    assert json.loads(definition_path.read_text(encoding="utf-8")) == {"local": "edit"}

    assert download_solexecbench.main([*args, "--force"]) == 0
    assert (
        json.loads(definition_path.read_text(encoding="utf-8"))["name"] == "matmul_demo"
    )


def test_verify_only_writes_manifest_without_downloading(tmp_path, monkeypatch):
    def fail_load_dataset(*args, **kwargs):
        raise AssertionError("verify-only should not download")

    monkeypatch.setattr(download_solexecbench, "load_dataset", fail_load_dataset)
    output_root = tmp_path / "dataset"
    problem_dir = output_root / "L1" / "matmul_demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text("{}\n", encoding="utf-8")
    (problem_dir / "workload.jsonl").write_text('{"uuid":"w"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    rc = download_solexecbench.main(
        [
            "--verify-only",
            "--category",
            "L1",
            "--output-root",
            str(output_root),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert rc == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["local_provenance"] == "verify_only_local_layout"
    assert manifest["claim_boundary"]["acquisition_or_layout_complete"] is True


def test_unknown_category_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown SOL-ExecBench category"):
        download_solexecbench.main(
            ["--category", "CUDA", "--output-root", str(tmp_path)]
        )


@pytest.mark.parametrize("unsafe_name", ["../escape", "/tmp/escape", "", ".", ".."])
def test_downloader_rejects_unsafe_remote_problem_names(
    tmp_path, monkeypatch, unsafe_name
):
    monkeypatch.setattr(
        download_solexecbench,
        "load_dataset",
        lambda *args, **kwargs: [_row(name=unsafe_name)],
    )

    rc = download_solexecbench.main(
        ["--category", "L1", "--output-root", str(tmp_path / "dataset")]
    )

    assert rc == 1
    assert not (tmp_path / "escape" / "definition.json").exists()


def test_download_data_shell_fails_fast():
    script = (REPO_ROOT / "scripts" / "download_data.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
