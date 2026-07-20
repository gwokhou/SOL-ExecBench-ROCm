from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from solar.analysis import orojenesis


def _matmul(
    activation: str,
    weight: str,
    output: str,
    *,
    m: int,
    k: int,
    n: int,
    batch: int | None = None,
) -> dict:
    equation = "MK,KN->MN" if batch is None else "BMK,KN->BMN"
    input_shape = [m, k] if batch is None else [batch, m, k]
    output_shape = [m, n] if batch is None else [batch, m, n]
    return {
        "semantic_op": {
            "kind": "einsum",
            "equation": equation,
            "effects": {
                "mutates": False,
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {"inputs": [activation, weight], "outputs": [output]},
        "tensor_shapes": {
            "inputs": [input_shape, [k, n]],
            "outputs": [output_shape],
        },
        "tensor_dtypes": {
            "inputs": ["float16", "float16"],
            "outputs": ["float16"],
        },
    }


def _runner(tmp_path: Path) -> orojenesis.OrojenesisRunner:
    runner = object.__new__(orojenesis.OrojenesisRunner)
    runner.home = tmp_path
    runner.mapper = tmp_path / "timeloop-mapper"
    runner.timeout_seconds = 10
    runner.toolchain_identity = {"verification_mode": "test-double"}
    return runner


def _write_mapping(path: Path, *, tile: int, word_bytes: int = 2) -> None:
    row: list[object] = [0] * 24
    row[0] = 32 * tile
    row[2] = 6
    row[3] = f"mapping-m{tile}"
    row[5] = 120
    row[6] = 8
    row[10] = tile * word_bytes * 2
    row[11] = tile * word_bytes * 2
    row[21:24] = [2, 3, 1]
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerow(row)


def _mapping_subprocess(args, *, cwd, **kwargs):
    del kwargs
    assert "-o" in args
    workdir = Path(cwd)
    tile = int(workdir.name.rsplit("-", 1)[-1])
    _write_mapping(workdir / "timeloop-mapper.oaves.csv", tile=tile)
    return SimpleNamespace(returncode=0, stdout="mapper stdout", stderr="")


def test_run_layer_emits_auditable_evidence(tmp_path, monkeypatch):
    runner = _runner(tmp_path)

    def fake_run(args, *, cwd, **kwargs):
        del args, kwargs
        Path(cwd, "timeloop-mapper.oaves.csv").write_text(
            "64,1.0,12\n128,2.0,8\n", encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="stdout", stderr="stderr")

    monkeypatch.setattr(orojenesis.subprocess, "run", fake_run)
    result = runner.run_layer(
        _matmul("x", "w", "y", m=2, k=3, n=4),
        tmp_path / "layer",
        word_bits=16,
    )
    assert result["curve"][-1]["dram_bytes"] == 16
    assert result["toolchain"] == {"verification_mode": "test-double"}
    assert set(result["evidence_files"]) == {
        "problem.yaml",
        "architecture.yaml",
        "mapper.yaml",
        "curve",
    }
    assert (tmp_path / "layer" / "stdout.log").read_text() == "stdout"


@pytest.mark.parametrize("failure", ["raise", "returncode"])
def test_run_layer_reports_process_failures(tmp_path, monkeypatch, failure):
    runner = _runner(tmp_path)

    def fake_run(*args, **kwargs):
        del args, kwargs
        if failure == "raise":
            raise OSError("cannot execute")
        return SimpleNamespace(returncode=7, stdout="", stderr="failed")

    monkeypatch.setattr(orojenesis.subprocess, "run", fake_run)
    message = "execution failed" if failure == "raise" else "status 7"
    with pytest.raises(orojenesis.OrojenesisError, match=message):
        runner.run_layer(
            _matmul("x", "w", "y", m=2, k=3, n=4),
            tmp_path / failure,
            word_bits=16,
        )


def test_run_multi_chain_composes_sweeps(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(orojenesis.subprocess, "run", _mapping_subprocess)
    chain = [
        ("mm0", _matmul("x", "w0", "hidden", m=2, k=3, n=4)),
        ("mm1", _matmul("hidden", "w1", "result", m=2, k=4, n=5)),
    ]
    result = runner.run_multi_chain(chain, tmp_path / "chain", word_bits=16)
    assert result["composition"] == orojenesis.MULTI_EINSUM_COMPOSITION
    assert [item["role"] for item in result["sweeps"]] == [
        "first",
        "second_last",
    ]
    assert {item["row_tile"] for item in result["curve"]} == {1}
    parsed = orojenesis.parse_multi_einsum_curve(
        tmp_path / "chain" / "multi-einsum-curve.csv", word_bytes=2
    )
    assert parsed == result["curve"]


def test_run_multi_chain_validates_width_and_process(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    chain = [
        ("mm0", _matmul("x", "w0", "hidden", m=2, k=3, n=4)),
        ("mm1", _matmul("hidden", "w1", "result", m=2, k=4, n=5)),
    ]
    with pytest.raises(orojenesis.OrojenesisError, match="byte aligned"):
        runner.run_multi_chain(chain, tmp_path / "width", word_bits=7)

    monkeypatch.setattr(
        orojenesis.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=9, stdout="", stderr="failure"
        ),
    )
    with pytest.raises(orojenesis.OrojenesisError, match="status 9"):
        runner.run_multi_chain(chain, tmp_path / "failed", word_bits=16)


def _batched_region() -> dict:
    first_layer = _matmul("x", "w0", "hidden", m=3, k=4, n=5, batch=2)
    second_layer = _matmul("hidden", "w1", "result", m=3, k=5, n=6, batch=2)
    first = orojenesis._region_matmul_descriptor("mm0", first_layer)
    second = orojenesis._region_matmul_descriptor("mm1", second_layer)
    return {
        "schema_version": 1,
        "kind": "broadcast_batch_linear_matmul",
        "composition": orojenesis.MULTI_EINSUM_BATCH_COMPOSITION,
        "nodes": [first, second],
        "edges": [
            {
                "producer": "mm0",
                "consumer": "mm1",
                "tensor": "hidden",
                "bridges": [],
                "axis_map": [0, 1],
                "layer_path": ["mm0", "mm1"],
            }
        ],
        "roots": ["mm0"],
        "leaves": ["mm1"],
        "schedule": ["mm0", "mm1"],
        "physical_paths": [["mm0", "mm1"]],
    }


def test_run_multi_region_composes_sweeps(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(orojenesis.subprocess, "run", _mapping_subprocess)
    result = runner.run_multi_region(
        _batched_region(), tmp_path / "region", word_bits=16
    )
    assert result["composition"] == orojenesis.MULTI_EINSUM_BATCH_COMPOSITION
    assert len(result["sweeps"]) == 2
    assert result["curve"]
    assert (
        orojenesis.parse_multi_einsum_region_curve(
            tmp_path / "region" / "multi-einsum-region-curve.csv", word_bytes=2
        )
        == result["curve"]
    )


def test_run_multi_region_reports_process_failures(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    with pytest.raises(orojenesis.OrojenesisError, match="byte aligned"):
        runner.run_multi_region(_batched_region(), tmp_path / "width", word_bits=0)

    def fake_run(*args, **kwargs):
        del args, kwargs
        raise subprocess.SubprocessError("failed")

    monkeypatch.setattr(orojenesis.subprocess, "run", fake_run)
    with pytest.raises(orojenesis.OrojenesisError, match="execution failed"):
        runner.run_multi_region(_batched_region(), tmp_path / "failed", word_bits=16)
