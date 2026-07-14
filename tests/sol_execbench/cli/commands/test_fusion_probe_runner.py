# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

from importlib import resources
from pathlib import Path

from sol_execbench.cli.commands import fusion_probe_runner
from sol_execbench_type_helpers import make_definition, make_workload


def test_fusion_probe_sources_are_package_resources() -> None:
    source_resources = resources.files("sol_execbench.data").joinpath("fusion_probes")

    with resources.as_file(source_resources) as source_dir:
        assert (source_dir / "fusion_probe_runner.hip").is_file()
        assert (source_dir / "gfx1200_reduction_epilogue.hip").is_file()


def test_compile_runner_uses_materialized_resource_directory(
    tmp_path: Path, monkeypatch
) -> None:
    commands: list[tuple[str, ...]] = []
    source_dir = tmp_path / "resources"
    source_dir.mkdir()
    (source_dir / "fusion_probe_runner.hip").write_text("probe", encoding="utf-8")
    monkeypatch.setattr(
        fusion_probe_runner.subprocess,
        "run",
        lambda command, **_: (
            commands.append(command)
            or type("Result", (), {"returncode": 0, "stderr": ""})()
        ),
    )

    executable, command = fusion_probe_runner._compile_runner(
        tmp_path, "gfx1200", source_dir
    )

    assert executable == tmp_path / "fusion_probe_runner"
    assert command == commands[0]
    assert f"-I{source_dir}" in command
    assert str(source_dir / "fusion_probe_runner.hip") in command


def test_probe_signature_is_derived_from_exact_bound_graph_group() -> None:
    definition = make_definition(
        name="sum_then_scalar",
        axes={
            "B": {"type": "const", "value": 2},
            "S": {"type": "const", "value": 3},
            "H": {"type": "const", "value": 4},
        },
        inputs={
            "x": {"shape": ["B", "S", "H"], "dtype": "float32"},
            "scalar": {"shape": [], "dtype": "float32"},
        },
        outputs={"out": {"shape": [], "dtype": "float32"}},
        reference="def run(x, scalar):\n    return torch.sum(x) * scalar\n",
    )
    workload = make_workload(
        axes={},
        inputs={"x": {"type": "random"}, "scalar": {"type": "random"}},
        uuid="sum-then-scalar",
    )

    signature = fusion_probe_runner._workload_fusion_signature(
        definition,
        workload,
        architecture="gfx1200",
        kind="tanh",
    )

    assert signature.pattern_id == "diagnostic_reduction_epilogue.v1"
    assert signature.op_names == ("torch.sum", "torch.mul")
    assert signature.input_shapes == ((), (2, 3, 4))
    assert signature.output_shapes == ((),)
    assert signature.tile_contract == {
        "workgroup_size": 256,
        "reduction": "sum",
        "epilogue": "scalar_mul",
    }
