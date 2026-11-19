"""Execute the separately frozen, development-only LayerNorm GPU acceptance."""

import argparse
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

from scripts.internal.solar.e2_contracts import (
    PRIVATE_FILE,
    Activity,
    Mode,
    Session,
)
from scripts.internal.solar.e2_evidence_budget import retained_evidence
from scripts.internal.solar.e2_layernorm_dev_inputs import (
    FEATURE_SHIFTS,
    instrument_development,
)
from scripts.internal.solar.e2_runner import _evaluate_staged
from scripts.internal.solar.e2_staging import (
    FACTORY_NAME,
    LayerNormReference,
    package,
    problem,
)
from scripts.internal.solar.e2_successor_exposure import (
    protect_observation_process,
)
from scripts.internal.solar.e2_successor_runner import (
    cold_session_workspace,
    require_empty_scratch,
)
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    verify_clock_owner,
)
from sol_execbench.core.bench.gpu_execution import (
    native_gpu_execution,
    resolve_gpu_device,
)
from sol_execbench.core.bench.gpu_lock_paths import configured_gpu_locks
from sol_execbench.core.bench.judge.candidate_catalog import candidate_source
from sol_execbench.core.bench.judge.contracts import (
    DEVELOPMENT_INSTANCES,
    REPETITIONS,
    TASK_FAMILIES,
    WORKLOAD_INDICES,
    JudgeCase,
    JudgeFamily,
    JudgeInstance,
    JudgeStrategy,
    JudgeTask,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_bytes, sha256_file
from sol_execbench.core.process.environment import (
    ENV_SOL_EXECBENCH_DEVICE,
    ENV_SOL_EXECBENCH_INPUT_NONCE,
    ENV_SOL_EXECBENCH_SANDBOXED,
)
from sol_execbench.core.scoring.release_environment import (
    current_release_execution_identity,
    verify_release_source_state,
)
from solar.verification.execution import require_current_semantic_runtime

PROTOCOL = "layernorm-centered-fp64-development-v1"
REFERENCE = LayerNormReference.CENTERED_FP64_V1
GPU_SECONDS = 3600
SESSION_SECONDS = 120
ORACLE_SECONDS_PER_SESSION = 120
EVIDENCE_BYTES = 6 * 1024**3
SESSION_EVIDENCE_RESERVE = 1024**3 + 16 * 1024**2
TEARDOWN_SECONDS = 60
CORRECTNESS_CALLS = 10
COMPLETE_CALLS = CORRECTNESS_CALLS + 1
ACTIVITY_DIRECTORY = "development"


@dataclass(frozen=True, slots=True, kw_only=True)
class Slot:
    """A frozen slot with shared generation coordinates and an explicit rotation."""

    ordinal: int
    session: Session
    shift: int
    lane: str

    @property
    def name(self) -> str:
        """Use the slot ordinal even for repeated or early-failed sessions."""
        return f"session-{self.ordinal:03d}"

    @property
    def nonce(self) -> str:
        """Reuse only development input derivation, across separate channels."""
        case = self.session.case
        key = f"{PROTOCOL}/{case.logical_id}"
        if self.lane == "negative":
            key += f"/negative/{self.session.repeat}"
        return sha256_bytes(key.encode())

    @property
    def uuid(self) -> str:
        """Keep the input derivation UUID fixed across positive comparisons."""
        return f"{PROTOCOL}-{self.session.case.workload_index}-{self.session.case.family}"

    def metadata(self) -> dict[str, object]:
        """Record the complete denominator, even when a slot is not reached."""
        return {
            **self.session.metadata(),
            "ordinal": self.ordinal,
            "lane": self.lane,
            "feature_shift": self.shift,
            "staged_workload_uuid": self.uuid,
        }


def _cases() -> list[tuple[Session, int, str]]:
    cases = []
    task = JudgeTask.LAYERNORM
    for index in WORKLOAD_INDICES:
        for family in TASK_FAMILIES[task]:
            case = JudgeCase(
                task=task,
                workload_index=index,
                family=family,
                instance=JudgeInstance.DEVELOPMENT_0,
            )
            for strategy in (
                JudgeStrategy.REFERENCE,
                JudgeStrategy.ALTERNATIVE,
            ):
                for repeat, shift in enumerate((0, 0, *FEATURE_SHIFTS)):
                    session = Session(
                        case=case,
                        strategy=strategy,
                        repeat=repeat,
                        mode=Mode.DIAGNOSTIC,
                        activity=Activity.CORRECTNESS,
                    )
                    cases.append((session, shift, "control"))
        for family, strategy in (
            (JudgeFamily.L3, JudgeStrategy.SEMANTIC),
            (JudgeFamily.L0, JudgeStrategy.NUMERICAL),
        ):
            for instance in DEVELOPMENT_INSTANCES:
                case = JudgeCase(
                    task=task,
                    workload_index=index,
                    family=family,
                    instance=instance,
                )
                for repeat in range(REPETITIONS):
                    session = Session(
                        case=case,
                        strategy=strategy,
                        repeat=repeat,
                        mode=Mode.DIAGNOSTIC,
                        activity=Activity.CORRECTNESS,
                    )
                    cases.append((session, 0, "negative"))
    return cases


def roster() -> tuple[Slot, ...]:
    """Materialize only metadata for the 120 controls and 36 negative slots."""
    return tuple(
        Slot(ordinal=i, session=session, shift=shift, lane=lane)
        for i, (session, shift, lane) in enumerate(_cases())
    )


def stage_identities(repository: Path) -> dict[str, Any]:
    """Bind the staged reference, controls, original checks and candidate source."""
    bindings = {}
    for slot in roster():
        session = slot.session
        case = session.case
        key = f"{case.workload_index}/{case.family}/{session.strategy}"
        if key in bindings:
            continue
        definition, workload = problem(
            repository, session, slot.uuid, layernorm_reference=REFERENCE
        )
        source = (
            definition.reference.split(f"\n\ndef {FACTORY_NAME}(", 1)[0]
            if session.strategy is JudgeStrategy.REFERENCE
            else candidate_source(case.task, session.strategy)
        )
        bindings[key] = {
            "reference_sha256": sha256_bytes(definition.reference.encode()),
            "candidate_sha256": sha256_bytes(source.encode()),
            "checks": [c.model_dump(mode="json") for c in workload.checks],
        }
    return bindings


def verify_bundle(
    repository: Path, bundle: Path, expected_sha: str
) -> dict[str, Any]:
    """Verify all activity source identities and the selected runtime before use."""
    path = bundle / "freeze.json"
    if sha256_file(path) != expected_sha:
        raise ValueError("LayerNorm GPU freeze SHA-256 changed")
    spec = load_json_value(path)
    if spec["protocol"] != PROTOCOL or spec["reference"] != REFERENCE:
        raise ValueError("unexpected LayerNorm GPU protocol or reference")
    capsule = bundle / "capsule"
    actual = {
        file.relative_to(capsule).as_posix(): sha256_file(file)
        for file in capsule.rglob("*.py")
    }
    if actual != spec["capsule_sha256"]:
        raise ValueError("LayerNorm GPU research capsule changed")
    if (
        sha256_file(bundle / "protocol.md") != spec["protocol_sha256"]
        or sha256_file(bundle / "cpu-acceptance.json")
        != spec["cpu_acceptance_sha256"]
    ):
        raise ValueError("LayerNorm GPU evidence binding changed")
    verify_release_source_state(
        repository, expected_revision=spec["execution_commit"]
    )
    identity = current_release_execution_identity(spec["execution_commit"])
    if identity.container_image_id != spec["image_id"]:
        raise ValueError("LayerNorm GPU image changed")
    require_current_semantic_runtime(spec["container_semantic_runtime"])
    if (
        stage_identities(repository) != spec["staged_identities"]
        or [s.metadata() for s in roster()] != spec["roster"]
        or sorted(os.sched_getaffinity(0)) != spec["cpu_affinity"]
    ):
        raise ValueError(
            "LayerNorm GPU staging, roster or CPU affinity changed"
        )
    return spec


def _run_slot(
    repository: Path, output: Path, slot: Slot, remaining: float
) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    atomic_write_json_value(output / "started.json", slot.metadata())
    deadline = time.monotonic() + min(SESSION_SECONDS, remaining)
    env = {
        ENV_SOL_EXECBENCH_DEVICE: "cuda:0",
        ENV_SOL_EXECBENCH_INPUT_NONCE: slot.nonce,
    }
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        with (
            cold_session_workspace(output) as workspace,
            package(
                repository,
                output,
                slot.session,
                workload_uuid=slot.uuid,
                private_seed=slot.session.case.require_development(),
                layernorm_reference=REFERENCE,
            ) as staged,
        ):
            candidate = sha256_bytes(
                staged.solution.sources[0].content.encode()
            )
            reference = sha256_bytes(staged.definition.reference.encode())
            result = _evaluate_staged(
                staged,
                slot.session,
                output,
                device="cuda:0",
                start=time.monotonic(),
                deadline=deadline,
                instrument=partial(
                    instrument_development,
                    shift=slot.shift,
                    oracle_cpu_allowance_seconds=ORACLE_SECONDS_PER_SESSION,
                ),
            )
        result.update(
            **workspace,
            candidate_sha256=candidate,
            reference_sha256=reference,
            feature_shift=slot.shift,
        )
    finally:
        (output / PRIVATE_FILE).unlink(missing_ok=True)
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
    return result


def _owned_online(repository: Path, destination: Path, freeze_sha: str) -> None:
    locks = configured_gpu_locks()
    owner = verify_clock_owner(
        locks.directory, os.environ[ENV_CLOCK_OWNER], (resolve_gpu_device(),)
    )
    if not owner.container_supervised:
        raise ValueError(
            "LayerNorm acceptance requires a container-supervised owner"
        )
    destination.mkdir(exist_ok=False)
    atomic_write_json_value(
        destination / "online-owner.json", owner.model_dump(mode="json")
    )
    started = time.monotonic()
    stop = None
    for slot in roster():
        remaining = (
            GPU_SECONDS - TEARDOWN_SECONDS - (time.monotonic() - started)
        )
        used = retained_evidence(destination)
        if (
            remaining < 1
            or max(used.logical_bytes, used.allocated_bytes)
            + SESSION_EVIDENCE_RESERVE
            > EVIDENCE_BYTES
        ):
            stop = "frozen_wall_or_evidence_budget"
            break
        output = destination / slot.name
        result = _run_slot(repository, output, slot, remaining)
        atomic_write_json_value(output / "session.json", result)
        if result["stop_reason"]:
            stop = result["stop_reason"]
            break
    atomic_write_json_value(
        destination / "online-summary.json",
        {
            "freeze_sha256": freeze_sha,
            "stop_reason": stop,
            "wall_seconds": time.monotonic() - started,
            "performance_authority": False,
        },
    )


def main() -> None:
    """Run the sole development entry after verifying its frozen identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    spec = verify_bundle(args.repository, args.bundle, args.expected_sha)
    if (
        os.environ.get(ENV_SOL_EXECBENCH_SANDBOXED) != "1"
        or not configured_gpu_locks().host_managed
    ):
        raise ValueError(
            "LayerNorm validation requires the hardened native owner"
        )
    require_empty_scratch()
    if sha256_file(Path("/proc/1/exe")) != spec["docker_init"]["sha256"]:
        raise ValueError("LayerNorm GPU Docker init differs from the freeze")
    with native_gpu_execution(manage_clocks=True):
        physical = resolve_gpu_device()
        if (physical.uuid, physical.bdf) != (
            spec["device"]["uuid"],
            spec["device"]["bdf"],
        ):
            raise ValueError("LayerNorm acceptance targeted another GPU")
        dev = torch.cuda.get_device_properties(0)
        if (
            torch.cuda.device_count() != 1
            or dev.gcnArchName.split(":")[0] != "gfx1200"
        ):
            raise ValueError(
                "LayerNorm acceptance runtime does not match gfx1200"
            )
        protect_observation_process()
        _owned_online(
            args.repository, args.bundle / ACTIVITY_DIRECTORY, args.expected_sha
        )


if __name__ == "__main__":
    main()
