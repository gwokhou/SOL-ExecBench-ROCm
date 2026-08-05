from __future__ import annotations

import csv
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from solar.analysis import orojenesis
from solar.analysis.orojenesis import (
    multi_einsum,
    process as orojenesis_process,
    runner as orojenesis_runner,
)
from solar.schema_versions import (
    OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION,
)


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


def _conv_proof() -> dict:
    layer = _matmul("x", "w", "y", m=2, k=3, n=4)
    layer["semantic_op"] = {
        "kind": "einsum",
        "target": "einsum",
        "equation": "BC(P+R)(Q+S),OCRS->BOPQ",
        "proof_source": {"kind": "aten", "target": "conv2d"},
        "effects": {
            "mutates": [],
            "aliases": [],
            "atomic": False,
            "opaque_library_call": False,
        },
    }
    layer["tensor_shapes"] = {
        "inputs": [[1, 2, 4, 4], [3, 2, 3, 3]],
        "outputs": [[1, 3, 2, 2]],
    }
    return layer


def _bmm_proof() -> dict:
    layer = _matmul("x", "w", "y", m=2, k=3, n=4)
    layer["semantic_op"] = {
        "kind": "einsum",
        "target": "einsum",
        "equation": "BMK,BKN->BMN",
        "proof_source": {"kind": "aten", "target": "bmm"},
        "effects": {
            "mutates": [],
            "aliases": [],
            "atomic": False,
            "opaque_library_call": False,
        },
    }
    layer["tensor_shapes"] = {
        "inputs": [[2, 2, 3], [2, 3, 4]],
        "outputs": [[2, 2, 4]],
    }
    return layer


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


def _write_witness_outputs(
    output: Path,
    *,
    buffer_capacity: tuple[int, int, int] = (32, 54, 12),
    element_counts: tuple[int, int, int] = (32, 54, 12),
) -> None:
    def vector(name: str, values: tuple[int, int, int]) -> str:
        items = "".join(f"<item>{value}</item>" for value in values)
        return f"<{name}><PerDataSpace>{items}</PerDataSpace></{name}>"

    def level(
        name: str,
        *,
        capacity: tuple[int, int, int],
        reads: tuple[int, int, int],
        updates: tuple[int, int, int],
    ) -> str:
        stats = "".join(
            (
                vector("keep", (1, 1, 1)),
                vector("utilized_capacity", capacity),
                vector("utilized_instances", (1, 1, 1)),
                vector("reads", reads),
                vector("updates", updates),
                vector("fills", (0, 0, 0)),
            ),
        )
        return (
            "<item><px><specs_><LevelSpecs>"
            f"<level_name>{name}</level_name>"
            "</LevelSpecs><word_bits><t_>32</t_></word_bits></specs_>"
            f"<stats_>{stats}</stats_></px></item>"
        )

    xml = "".join(
        (
            "<boost_serialization><engine><topology_><levels_>",
            level(
                "Buffer",
                capacity=buffer_capacity,
                reads=(0, 0, 0),
                updates=(0, 0, 0),
            ),
            level(
                "MainMemory",
                capacity=element_counts,
                reads=(element_counts[0], element_counts[1], 0),
                updates=(0, 0, element_counts[2]),
            ),
            "</levels_></topology_></engine></boost_serialization>",
        ),
    )
    (output / "timeloop-mapper.map+stats.xml").write_text(xml)
    for name in (
        "timeloop-mapper.map.txt",
        "timeloop-mapper.stats.txt",
        "timeloop-mapper.map.yaml",
    ):
        (output / name).write_text("fixed witness\n")


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
            "64,1.0,12\n128,2.0,8\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="stdout", stderr="stderr")

    monkeypatch.setattr(orojenesis_runner, "_default_mapper_runner", fake_run)
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
    assert not (tmp_path / "layer" / "stdout.log").exists()
    assert not (tmp_path / "layer" / "stderr.log").exists()


def test_run_layer_certifies_a_selected_capacity_compulsory_witness(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path)
    observed: dict[str, object] = {}

    def fake_run(args, *, cwd, **kwargs):
        del args
        observed.update(kwargs)
        row: list[object] = [0] * 18
        row[0] = 392
        row[1] = 1.0
        row[2] = 98
        row[3] = "fixed-compulsory-witness"
        row[-3:] = [32, 54, 12]
        with Path(cwd, "timeloop-mapper.oaves.csv").open(
            "w",
            newline="",
        ) as handle:
            csv.writer(handle).writerow(row)
        _write_witness_outputs(Path(cwd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orojenesis_runner, "_default_mapper_runner", fake_run)
    result = runner.run_layer(
        _conv_proof(),
        tmp_path / "conv",
        word_bits=32,
        selected_capacity_bytes=1024,
    )

    environment = cast(Mapping[str, object], observed["env"])
    assert environment["TIMELOOP_ENABLE_FIRST_READ_ELISION"] == "1"
    assert result["optimality_certificate"] == {
        "kind": "selected_capacity_compulsory_witness_v1",
        "scope": "selected_capacity_only",
        "pareto_curve_complete": False,
        "capacity_bytes": 1024,
        "buffer_bytes": 392,
        "compulsory_accesses_words": 98,
        "data_space_accesses_words": {
            "Input0": 32,
            "Input1": 54,
            "Output": 12,
        },
        "main_memory_accesses_words": {
            "reads": {"Input0": 32, "Input1": 54, "Output": 0},
            "updates": {"Input0": 0, "Input1": 0, "Output": 12},
            "fills": {"Input0": 0, "Input1": 0, "Output": 0},
        },
        "buffer_utilized_capacity_words": {
            "Input0": 32,
            "Input1": 54,
            "Output": 12,
        },
        "buffer_utilized_instances": {
            "Input0": 1,
            "Input1": 1,
            "Output": 1,
        },
        "proof": "feasible traffic equals the universal compulsory lower bound",
        "portability": {
            "compulsory_lower_bound": "architecture_independent",
            "achievability": "selected_architecture_cache_only",
        },
        "theorem_inputs": {
            "proof_source": {"kind": "aten", "target": "conv2d"},
            "equation": "BC(P+R)(Q+S),OCRS->BOPQ",
            "tensor_shapes": {
                "inputs": [[1, 2, 4, 4], [3, 2, 3, 3]],
                "outputs": [[1, 3, 2, 2]],
            },
            "data_spaces": ["Input0", "Input1", "Output"],
            "word_bits": 32,
            "dense": True,
            "first_read_elision": True,
            "instances": 1,
        },
        "streaming_axis": "B",
        "streaming_dimension": "A",
    }
    assert "environment.yaml" in result["evidence_files"]
    assert "timeloop-mapper.map+stats.xml" in result["evidence_files"]
    mapper = (tmp_path / "conv" / "mapper.yaml").read_text()
    assert "A=1 B=2 C=3 D=3 E=3 F=2 G=2" in mapper
    architecture = (tmp_path / "conv" / "architecture.yaml").read_text()
    assert "sizeKB: 1" in architecture


def test_run_layer_rejects_a_noncompulsory_convolution_witness(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path)

    def fake_run(args, *, cwd, **kwargs):
        del args, kwargs
        row: list[object] = [0] * 18
        row[0:4] = [392, 1.0, 99, "invalid-witness"]
        row[-3:] = [33, 54, 12]
        with Path(cwd, "timeloop-mapper.oaves.csv").open(
            "w",
            newline="",
        ) as handle:
            csv.writer(handle).writerow(row)
        _write_witness_outputs(Path(cwd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orojenesis_runner, "_default_mapper_runner", fake_run)
    with pytest.raises(
        orojenesis.OrojenesisError,
        match="did not reach the selected-capacity optimum",
    ):
        runner.run_layer(
            _conv_proof(),
            tmp_path / "invalid-conv",
            word_bits=32,
            selected_capacity_bytes=1024,
        )


def test_run_layer_certifies_a_batched_matmul_compulsory_witness(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path)

    def fake_run(args, *, cwd, **kwargs):
        del args, kwargs
        row: list[object] = [0] * 18
        row[0:4] = [104, 1.0, 52, "fixed-bmm-compulsory-witness"]
        row[-3:] = [12, 24, 16]
        with Path(cwd, "timeloop-mapper.oaves.csv").open(
            "w",
            newline="",
        ) as handle:
            csv.writer(handle).writerow(row)
        _write_witness_outputs(
            Path(cwd),
            buffer_capacity=(6, 12, 8),
            element_counts=(12, 24, 16),
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orojenesis_runner, "_default_mapper_runner", fake_run)
    result = runner.run_layer(
        _bmm_proof(),
        tmp_path / "bmm",
        word_bits=32,
        selected_capacity_bytes=1024,
    )

    certificate = result["optimality_certificate"]
    assert certificate["compulsory_accesses_words"] == 52
    assert certificate["buffer_utilized_capacity_words"] == {
        "Input0": 6,
        "Input1": 12,
        "Output": 8,
    }
    assert certificate["theorem_inputs"]["proof_source"] == {
        "kind": "aten",
        "target": "bmm",
    }
    assert certificate["streaming_dimension"] == "A"
    mapper = (tmp_path / "bmm" / "mapper.yaml").read_text()
    assert "A=1 B=2 C=3 D=4" in mapper
    assert "A=2 B=1 C=1 D=1" in mapper


def test_batched_matmul_witness_falls_back_when_slice_exceeds_capacity(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path)

    def fake_run(args, *, cwd, **kwargs):
        del args, kwargs
        Path(cwd, "timeloop-mapper.oaves.csv").write_text("64,1.0,52\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orojenesis_runner, "_default_mapper_runner", fake_run)
    result = runner.run_layer(
        _bmm_proof(),
        tmp_path / "large-bmm",
        word_bits=32,
        selected_capacity_bytes=100,
    )

    assert "optimality_certificate" not in result
    assert "environment.yaml" not in result["evidence_files"]


def test_selected_capacity_witness_rejects_a_spoofed_conv_equation(
    tmp_path,
    monkeypatch,
):
    runner = _runner(tmp_path)
    layer = _conv_proof()
    layer["semantic_op"]["equation"] = "BC(P+R)(Q+S),OCRS->BOQP"

    def fake_run(args, *, cwd, **kwargs):
        del args, kwargs
        Path(cwd, "timeloop-mapper.oaves.csv").write_text("392,1.0,98\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(orojenesis_runner, "_default_mapper_runner", fake_run)
    result = runner.run_layer(
        layer,
        tmp_path / "spoofed-conv",
        word_bits=32,
        selected_capacity_bytes=1024,
    )

    assert "optimality_certificate" not in result
    assert "environment.yaml" not in result["evidence_files"]


@pytest.mark.parametrize("failure", ["raise", "returncode"])
def test_run_layer_reports_process_failures(tmp_path, monkeypatch, failure):
    runner = _runner(tmp_path)

    def fake_run(*args, **kwargs):
        del args, kwargs
        if failure == "raise":
            raise OSError("cannot execute")
        return SimpleNamespace(returncode=7, stdout="", stderr="failed")

    monkeypatch.setattr(orojenesis_runner, "_default_mapper_runner", fake_run)
    message = "execution failed" if failure == "raise" else "status 7"
    with pytest.raises(orojenesis.OrojenesisError, match=message):
        runner.run_layer(
            _matmul("x", "w", "y", m=2, k=3, n=4),
            tmp_path / failure,
            word_bits=16,
        )


def test_run_multi_chain_composes_sweeps(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(
        orojenesis_runner,
        "_default_mapper_runner",
        _mapping_subprocess,
    )
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
        tmp_path / "chain" / "multi-einsum-curve.csv",
        word_bytes=2,
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
        orojenesis_runner,
        "_default_mapper_runner",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=9,
            stdout="",
            stderr="failure",
        ),
    )
    with pytest.raises(orojenesis.OrojenesisError, match="status 9"):
        runner.run_multi_chain(chain, tmp_path / "failed", word_bits=16)


def _batched_region() -> dict:
    first_layer = _matmul("x", "w0", "hidden", m=3, k=4, n=5, batch=2)
    second_layer = _matmul("hidden", "w1", "result", m=3, k=5, n=6, batch=2)
    first = multi_einsum._region_matmul_descriptor("mm0", first_layer)
    second = multi_einsum._region_matmul_descriptor("mm1", second_layer)
    return {
        "schema_version": OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION,
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
            },
        ],
        "roots": ["mm0"],
        "leaves": ["mm1"],
        "schedule": ["mm0", "mm1"],
        "physical_paths": [["mm0", "mm1"]],
    }


def test_multi_einsum_region_rejects_coerced_schema_version() -> None:
    region = _batched_region()
    region["schema_version"] = str(
        OROJENESIS_MULTI_EINSUM_REGION_SCHEMA_VERSION,
    )

    with pytest.raises(orojenesis.OrojenesisError, match="unsupported.*schema"):
        orojenesis.multi_einsum_region_problem(region)


def test_run_multi_region_composes_sweeps(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(
        orojenesis_runner,
        "_default_mapper_runner",
        _mapping_subprocess,
    )
    result = runner.run_multi_region(
        _batched_region(),
        tmp_path / "region",
        word_bits=16,
    )
    assert result["composition"] == orojenesis.MULTI_EINSUM_BATCH_COMPOSITION
    assert len(result["sweeps"]) == 2
    assert result["curve"]
    assert (
        orojenesis.parse_multi_einsum_region_curve(
            tmp_path / "region" / "multi-einsum-region-curve.csv",
            word_bytes=2,
        )
        == result["curve"]
    )


def test_run_multi_region_reports_process_failures(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    with pytest.raises(orojenesis.OrojenesisError, match="byte aligned"):
        runner.run_multi_region(
            _batched_region(),
            tmp_path / "width",
            word_bits=0,
        )

    def fake_run(*args, **kwargs):
        del args, kwargs
        raise subprocess.SubprocessError("failed")

    monkeypatch.setattr(orojenesis_runner, "_default_mapper_runner", fake_run)
    with pytest.raises(orojenesis.OrojenesisError, match="execution failed"):
        runner.run_multi_region(
            _batched_region(),
            tmp_path / "failed",
            word_bits=16,
        )


@pytest.mark.requires_linux
def test_default_mapper_runner_kills_timed_out_process_group(tmp_path):
    command = [
        sys.executable,
        "-c",
        (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
            "print(child.pid,flush=True); "
            "time.sleep(30)"
        ),
    ]

    with pytest.raises(subprocess.TimeoutExpired):
        orojenesis_runner._default_mapper_runner(
            command,
            cwd=tmp_path,
            timeout=0.2,
        )

    child_pid = int((tmp_path / "stdout.log").read_text().strip())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and _process_is_running(child_pid):
        time.sleep(0.05)
    assert not _process_is_running(child_pid)


@pytest.mark.requires_linux
def test_default_mapper_runner_sanitizes_environment_and_logs(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SOLAR_TEST_API_TOKEN", "environment-secret")
    command = [
        sys.executable,
        "-c",
        (
            "import os; "
            "print(os.getenv('SOLAR_TEST_API_TOKEN', 'not-inherited')); "
            "print('AUTHORIZATION: Bearer logged-secret')"
        ),
    ]

    completed = orojenesis_runner._default_mapper_runner(
        command,
        cwd=tmp_path,
        timeout=2,
    )

    assert completed.returncode == 0
    stdout = (tmp_path / "stdout.log").read_text()
    assert "not-inherited" in stdout
    assert "environment-secret" not in stdout
    assert "logged-secret" not in stdout
    assert "AUTHORIZATION: Bearer <redacted>" in stdout


@pytest.mark.requires_linux
def test_default_mapper_runner_bounds_output_while_running(tmp_path):
    command = [
        sys.executable,
        "-c",
        (
            "import os; "
            f"chunk=b'x'*{orojenesis_process.MAPPER_LOG_MAX_BYTES + 4096}; "
            "os.write(1, chunk)"
        ),
    ]

    completed = orojenesis_runner._default_mapper_runner(
        command,
        cwd=tmp_path,
        timeout=2,
    )

    assert completed.returncode == 0
    stdout = tmp_path / "stdout.log"
    assert stdout.stat().st_size <= orojenesis_process.MAPPER_LOG_MAX_BYTES
    assert stdout.read_bytes().startswith(b"[output truncated")


def _process_is_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"
