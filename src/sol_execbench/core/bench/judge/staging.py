"""Prepare independent development problems for the existing trusted evaluator."""

import struct
import uuid
from pathlib import Path

import torch

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.judge.candidate_catalog import candidate_solution
from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeFamily,
    JudgeInstance,
    JudgeStrategy,
    JudgeTask,
    load_case_problem,
)
from sol_execbench.core.bench.judge.inputs import generate_seeded_inputs
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
)
from sol_execbench.core.data.workload import ScalarInput, Workload

_FACTORY_NAME = "judge_development_inputs"


def stage_development_case(
    repository: Path, output: Path, case: JudgeCase
) -> None:
    """Write evaluator inputs and candidate sources; do not run or allocate tensors."""
    seed = case.require_development()
    definition, workload = load_case_problem(repository, case)
    output.mkdir(parents=True, exist_ok=False)
    reference = definition.reference + (
        f"\n\ndef {_FACTORY_NAME}(axes_and_scalars, device):\n"
        "    from sol_execbench.core.bench.judge.staging import development_factory\n"
        f"    return development_factory({case.task.value!r}, {case.family.value!r}, "
        f"{case.instance.value!r}, {case.workload_index}, axes_and_scalars, device)\n"
    )
    derived = Definition.model_validate(
        {
            **definition.model_dump(mode="json"),
            "reference": reference,
            "custom_inputs_entrypoint": _FACTORY_NAME,
        }
    )
    inputs = {
        name: (
            spec.model_dump(mode="json")
            if isinstance(spec, ScalarInput)
            else {"type": "custom"}
        )
        for name, spec in workload.inputs.items()
    }
    if case.task is JudgeTask.LAYERNORM:
        inputs["eps"] = {
            "type": "scalar",
            "value": struct.unpack("<f", struct.pack("<f", 1e-5))[0],
        }
    derived_workload = Workload.model_validate(
        {
            **workload.model_dump(mode="json"),
            "inputs": inputs,
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, case.logical_id)),
        }
    )
    config = BenchmarkConfig(
        seed=seed,
        warmup_runs=0,
        iterations=1,
        trials=1,
        min_measurement_time_seconds=None,
        lock_clocks=True,
        benchmark_reference=False,
    )
    atomic_write_json_value(
        output / "definition.json",
        derived.model_dump(mode="json"),
        sort_keys=False,
    )
    atomic_write_jsonl_values(output / "workload.jsonl", [derived_workload])
    atomic_write_json_value(
        output / "config.json", config.model_dump(mode="json")
    )
    for strategy in JudgeStrategy:
        solution = candidate_solution(case.task, strategy, derived.name)
        atomic_write_json_value(
            output / f"{strategy}.json", solution.model_dump(mode="json")
        )


def development_factory(
    task: str,
    family: str,
    instance: str,
    workload_index: int,
    axes_and_scalars: dict[str, int | float | bool],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Consume the trusted service's isolated invocation seed; expose no oracle."""
    case = JudgeCase(
        task=JudgeTask(task),
        family=JudgeFamily(family),
        instance=JudgeInstance(instance),
        workload_index=workload_index,
    )
    case.require_development()
    axes = {
        key: int(value)
        for key, value in axes_and_scalars.items()
        if key != "eps"
    }
    values = generate_seeded_inputs(
        case.task, case.family, axes, torch.initial_seed()
    ).values
    names = {
        JudgeTask.GEMM: ("A", "B"),
        JudgeTask.SUM: ("x",),
        JudgeTask.LAYERNORM: ("x", "weight", "bias"),
    }[case.task]
    return {
        name: value.to(device)
        for name, value in zip(names, values[: len(names)], strict=True)
    }
