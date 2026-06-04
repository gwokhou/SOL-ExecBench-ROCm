from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dataset_failure_mode_matrix_documents_reuse_and_incomplete_states():
    text = (REPO_ROOT / "docs" / "analysis.md").read_text(encoding="utf-8")

    assert "### Dataset Reuse And Failure-Mode Matrix" in text
    for phrase in (
        "stale provenance",
        "selected ready subset",
        "missing derived sidecar",
        "forced rerun",
        "CLI timeout or nonzero exit",
        "no trace output",
        "derived_evidence_missing",
        "bounded diagnostics",
    ):
        assert phrase in text

    assert "CPU-safe" in text
    assert "Live ROCm execution" in text
    assert "ROCm/Docker environment" in text


def test_dataset_sharding_contract_documents_default_cli_boundary():
    text = (REPO_ROOT / "docs" / "analysis.md").read_text(encoding="utf-8")

    for phrase in (
        "sol_execbench.core.dataset.sharding",
        "original input",
        "ordinal",
        "shard-0000-of-0002",
        "one trace file ref per shard",
        "duplicate workloads or incomplete shards",
        "profiler-backed timing phases remain serial",
        "--phase derived",
        "CPU/I/O-only report generation",
    ):
        assert phrase in text
