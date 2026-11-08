"""Stage E2 research drivers, retaining all incumbent review and runtime guards."""

import struct
from dataclasses import replace
from pathlib import Path

from scripts.internal.solar.e2_contracts import PRIVATE_FILE, Activity, Session
from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.judge.candidate_catalog import candidate_solution
from sol_execbench.core.bench.judge.contracts import (
    JudgeInstance,
    JudgeTask,
    load_case_problem,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.workload import ScalarInput, Workload
from sol_execbench.driver.problem_packager import ProblemPackager

FACTORY_NAME = "judge_experiment_inputs"


def problem(
    repository: Path, session: Session, workload_uuid: str
) -> tuple[Definition, Workload]:
    """Bind the original checks while keeping exposure roles out of public metadata."""
    case = session.case
    definition, workload = load_case_problem(
        repository, replace(case, instance=JudgeInstance.DEVELOPMENT_0)
    )
    reference = definition.reference + (
        f"\n\ndef {FACTORY_NAME}(axes_and_scalars, device):\n"
        "    from scripts.internal.solar.e2_inputs import factory\n"
        f"    return factory({case.task.value!r}, {case.family.value!r}, axes_and_scalars, device)\n"
    )
    derived = Definition.model_validate(
        {
            **definition.model_dump(mode="json"),
            "reference": reference,
            "custom_inputs_entrypoint": FACTORY_NAME,
        }
    )
    inputs = {
        name: spec.model_dump(mode="json")
        if isinstance(spec, ScalarInput)
        else {"type": "custom"}
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
            "uuid": workload_uuid,
        }
    )
    return derived, derived_workload


def config(session: Session) -> BenchmarkConfig:
    """The public seed is neutral; only the private worker knows the base seed."""
    cost = session.activity is Activity.COST
    return BenchmarkConfig(
        seed=0,
        warmup_runs=10 if cost else 0,
        iterations=50 if cost else 1,
        trials=3 if cost else 1,
        min_measurement_time_seconds=None,
        lock_clocks=True,
        benchmark_reference=cost,
    )


def package(
    repository: Path,
    output: Path,
    session: Session,
    *,
    workload_uuid: str,
    private_seed: int,
) -> ProblemPackager:
    """Prepare a one-shot evaluation, with private metadata removed before READY."""
    definition, workload = problem(repository, session, workload_uuid)
    packager = ProblemPackager(
        definition,
        [workload],
        candidate_solution(
            session.case.task, session.strategy, definition.name
        ),
        config(session),
        output,
        keep_output_dir=True,
    )
    atomic_write_json_value(
        output / PRIVATE_FILE,
        {
            **session.metadata(),
            "base_seed": private_seed,
        },
    )
    (output / PRIVATE_FILE).chmod(0o600)
    return packager


def instrument_drivers(output: Path) -> None:
    """Apply exact, count-checked instrumentation to copies of the stock templates."""
    edits = {
        "eval_driver.py": (
            (
                "_source_review = review_solution_sources(",
                (
                    "import time as _judge_time\n"
                    "_judge_static_start = _judge_time.monotonic()\n"
                    "_source_review = review_solution_sources("
                ),
            ),
            (
                "if _source_review.blocked:",
                (
                    "_reference_client.static_review_seconds = _judge_time.monotonic() - _judge_static_start\n"
                    "if _source_review.blocked:"
                ),
            ),
            (
                "_check_integrity = check_runtime_integrity",
                "_check_integrity = _reference_client.check_integrity",
            ),
            (
                "_reference_client = connect_reference_worker(device=_device)",
                (
                    "from scripts.internal.solar.e2_client import connect_observer\n"
                    "_reference_client = connect_observer(device=_device)"
                ),
            ),
            (
                "timing_recorder=_timing_records.append,",
                (
                    "timing_recorder=_timing_records.append,\n"
                    "            correctness_observer=_reference_client.observe,"
                ),
            ),
            (
                "finally:\n    _reference_client.close()",
                (
                    "finally:\n    _reference_client.close()\n"
                    "    import resource\n"
                    "    from sol_execbench.core.data.json_utils import atomic_write_json_value\n"
                    "    atomic_write_json_value(STAGING_DIR / 'judge-client-cost.json', "
                    "{'reference_ipc_wall_seconds_including_oracle': _reference_client.ipc_seconds, 'runtime_integrity_seconds': _reference_client.integrity_seconds, 'static_review_seconds': _reference_client.static_review_seconds, 'candidate_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})"
                ),
            ),
        ),
        "reference_worker.py": (
            (
                "from sol_execbench.driver.reference_worker_api import serve_reference_worker",
                "from scripts.internal.solar.e2_service import serve_observed_worker as serve_reference_worker",
            ),
        ),
    }
    for name, replacements in edits.items():
        path = output / name
        source = path.read_text()
        for original, replacement in replacements:
            if source.count(original) != 1:
                raise ValueError(f"E2 template seam changed: {name}")
            source = source.replace(original, replacement)
        path.write_text(source)
