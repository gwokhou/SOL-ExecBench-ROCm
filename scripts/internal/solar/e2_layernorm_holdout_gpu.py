"""Execute or verify the frozen LayerNorm h0 confirmation activity."""

import argparse
import json
import os
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import torch

from scripts.internal.solar.e2_contracts import (
    PRIVATE_FILE,
    Activity,
    Mode,
    Session,
    base_seed,
)
from scripts.internal.solar.e2_evidence_budget import retained_evidence
from scripts.internal.solar.e2_layernorm_dev_inputs import FEATURE_SHIFTS
from scripts.internal.solar.e2_layernorm_holdout_inputs import (
    instrument_holdout,
)
from scripts.internal.solar.e2_runner import _evaluate_staged
from scripts.internal.solar.e2_staging import (
    FACTORY_NAME,
    LayerNormReference,
    package,
    problem,
)
from scripts.internal.solar.e2_successor_exposure import (
    COMMITMENT_FILE,
    _write_exclusive,
    protect_observation_process,
    verify_worker_commitment,
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
    canonical_json_bytes,
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

PROTOCOL = "layernorm-centered-fp64-holdout-v1"
REFERENCE = LayerNormReference.CENTERED_FP64_V1
GPU_SECONDS = 3600
SESSION_SECONDS = 120
ORACLE_SECONDS_PER_SESSION = 120
EVIDENCE_BYTES = 6 * 1024**3
SESSION_EVIDENCE_RESERVE = 1024**3 + 16 * 1024**2
TEARDOWN_SECONDS = 60
CORRECTNESS_CALLS = 10
COMPLETE_CALLS = CORRECTNESS_CALLS + 1
ACTIVITY_DIRECTORY = "holdout-confirmation"
CLOSED_KEY_FILE = "holdout-key-after-online.bin"
COMMITMENT_KIND = "e2-successor-holdout-commitment-v1"
_DRAW_REPEATS = (5, 6)


class HoldoutLane(StrEnum):
    """Frozen session lanes within the LayerNorm h0 confirmation matrix."""

    CONTROL = "control"
    CONTROL_DRAW = "control_draw"
    NEGATIVE = "negative"


@dataclass(frozen=True, slots=True, kw_only=True)
class HoldoutSlot:
    """Frozen holdout slot metadata without materializing any h0 inputs."""

    ordinal: int
    session: Session
    shift: int
    lane: HoldoutLane

    @property
    def name(self) -> str:
        """Use the slot ordinal even when a session terminates early."""
        return f"session-{self.ordinal:03d}"

    @property
    def nonce(self) -> str:
        """Share anchor nonces for fixed/permuted checks; isolate draws and negatives."""
        case = self.session.case
        key = f"{PROTOCOL}/{case.logical_id}"
        if self.lane is HoldoutLane.CONTROL_DRAW:
            key += f"/draw/{self.session.repeat}"
        elif self.lane is HoldoutLane.NEGATIVE:
            key += f"/negative/{self.session.repeat}"
        return sha256_bytes(key.encode())

    @property
    def uuid(self) -> str:
        """Keep staged workload UUIDs aligned with nonce scoping."""
        case = self.session.case
        base = f"{PROTOCOL}-{case.workload_index}-{case.family}"
        if self.lane is HoldoutLane.CONTROL_DRAW:
            return f"{base}-draw-{self.session.repeat}"
        if self.lane is HoldoutLane.NEGATIVE:
            return f"{base}-neg-{self.session.repeat}"
        return base

    def metadata(self) -> dict[str, object]:
        """Serialize complete slot coordinates for roster and audit checks."""
        return {
            **self.session.metadata(),
            "ordinal": self.ordinal,
            "lane": self.lane.value,
            "feature_shift": self.shift,
            "staged_workload_uuid": self.uuid,
        }


def _control_slots(index: int) -> list[tuple[Session, int, HoldoutLane]]:
    items: list[tuple[Session, int, HoldoutLane]] = []
    task = JudgeTask.LAYERNORM
    for family in TASK_FAMILIES[task]:
        case = JudgeCase(
            task=task,
            workload_index=index,
            family=family,
            instance=JudgeInstance.HOLDOUT,
        )
        for strategy in (JudgeStrategy.REFERENCE, JudgeStrategy.ALTERNATIVE):
            for repeat, shift in enumerate((0, 0, *FEATURE_SHIFTS)):
                session = Session(
                    case=case,
                    strategy=strategy,
                    repeat=repeat,
                    mode=Mode.DIAGNOSTIC,
                    activity=Activity.CORRECTNESS,
                )
                items.append((session, shift, HoldoutLane.CONTROL))
            for repeat in _DRAW_REPEATS:
                session = Session(
                    case=case,
                    strategy=strategy,
                    repeat=repeat,
                    mode=Mode.DIAGNOSTIC,
                    activity=Activity.CORRECTNESS,
                )
                items.append((session, 0, HoldoutLane.CONTROL_DRAW))
    return items


def _negative_slots(index: int) -> list[tuple[Session, int, HoldoutLane]]:
    items: list[tuple[Session, int, HoldoutLane]] = []
    task = JudgeTask.LAYERNORM
    for family, strategy in (
        (JudgeFamily.L3, JudgeStrategy.SEMANTIC),
        (JudgeFamily.L0, JudgeStrategy.NUMERICAL),
    ):
        case = JudgeCase(
            task=task,
            workload_index=index,
            family=family,
            instance=JudgeInstance.HOLDOUT,
        )
        for repeat in range(REPETITIONS):
            session = Session(
                case=case,
                strategy=strategy,
                repeat=repeat,
                mode=Mode.DIAGNOSTIC,
                activity=Activity.CORRECTNESS,
            )
            items.append((session, 0, HoldoutLane.NEGATIVE))
    return items


def roster() -> tuple[HoldoutSlot, ...]:
    """Enumerate all 168 control and 18 negative holdout slots."""
    cases = [
        item
        for index in WORKLOAD_INDICES
        for item in (*_control_slots(index), *_negative_slots(index))
    ]
    return tuple(
        HoldoutSlot(ordinal=i, session=session, shift=shift, lane=lane)
        for i, (session, shift, lane) in enumerate(cases)
    )


def roster_digest() -> str:
    """Return the canonical SHA-256 of the 186 holdout roster rows."""
    return sha256_bytes(canonical_json_bytes([s.metadata() for s in roster()]))


def stage_identities(repository: Path) -> dict[str, Any]:
    """Bind the staged reference, controls, checks, and candidate source."""
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
    """Verify every frozen source, artifact, roster, and runtime identity."""
    path = bundle / "freeze.json"
    if sha256_file(path) != expected_sha:
        raise ValueError("LayerNorm holdout freeze SHA-256 changed")
    spec = load_json_value(path)
    if spec["protocol"] != PROTOCOL or spec["reference"] != REFERENCE:
        raise ValueError("unexpected LayerNorm holdout protocol or reference")
    capsule = bundle / "capsule"
    actual = {
        file.relative_to(capsule).as_posix(): sha256_file(file)
        for file in capsule.rglob("*.py")
    }
    if actual != spec["capsule_sha256"]:
        raise ValueError("LayerNorm holdout research capsule changed")
    if (
        sha256_file(bundle / "protocol.md") != spec["protocol_sha256"]
        or sha256_file(bundle / "cpu-acceptance.json")
        != spec["cpu_acceptance_sha256"]
        or sha256_file(bundle / "dev-audit.json") != spec["dev_audit_sha256"]
    ):
        raise ValueError("LayerNorm holdout evidence binding changed")
    verify_release_source_state(
        repository, expected_revision=spec["execution_commit"]
    )
    identity = current_release_execution_identity(spec["execution_commit"])
    if identity.container_image_id != spec["image_id"]:
        raise ValueError("LayerNorm holdout image changed")
    require_current_semantic_runtime(spec["container_semantic_runtime"])
    if (
        stage_identities(repository) != spec["staged_identities"]
        or [s.metadata() for s in roster()] != spec["roster"]
        or roster_digest() != spec["roster_sha256"]
        or sorted(os.sched_getaffinity(0)) != spec["cpu_affinity"]
    ):
        raise ValueError(
            "LayerNorm holdout staging, roster or CPU affinity changed"
        )
    return spec


def _commit_holdout(
    destination: Path, spec: dict[str, Any], freeze_sha: str
) -> tuple[bytes, str]:
    roster_path = destination / "roster.json"
    commitment_path = destination / COMMITMENT_FILE
    closed_key_path = destination / CLOSED_KEY_FILE
    online_dir = destination / "online"
    if any(
        candidate.exists()
        for candidate in (
            roster_path,
            commitment_path,
            closed_key_path,
            online_dir,
        )
    ):
        raise FileExistsError(
            "LayerNorm holdout commitment state already exists"
        )
    _write_exclusive(roster_path, canonical_json_bytes(spec["roster"]))
    if sha256_file(roster_path) != spec["roster_sha256"]:
        raise ValueError("LayerNorm holdout roster digest mismatch")
    key = secrets.token_bytes(32)
    _write_exclusive(
        commitment_path,
        canonical_json_bytes(
            {
                "research_format": COMMITMENT_KIND,
                "protocol": PROTOCOL,
                "key_sha256": sha256_bytes(key),
                "freeze_sha256": freeze_sha,
                "roster_sha256": spec["roster_sha256"],
                "schedule_sha256": spec["roster_sha256"],
                "instances_materialized_at_commitment": 0,
                "created_unix": time.time(),
                "performance_authority": False,
            }
        ),
    )
    return key, sha256_file(commitment_path)


def _run_slot(
    repository: Path,
    slot_dir: Path,
    slot: HoldoutSlot,
    remaining: float,
    *,
    key: bytes,
    commitment_sha: str,
) -> dict[str, Any]:
    slot_dir.mkdir(exist_ok=False)
    atomic_write_json_value(slot_dir / "started.json", slot.metadata())
    evidence_dir = slot_dir / "evidence"
    verify_worker_commitment(evidence_dir, commitment_sha)
    evidence_dir.mkdir(exist_ok=False)
    deadline = time.monotonic() + min(SESSION_SECONDS, remaining)
    env = {
        ENV_SOL_EXECBENCH_DEVICE: "cuda:0",
        ENV_SOL_EXECBENCH_INPUT_NONCE: slot.nonce,
    }
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        with (
            cold_session_workspace(evidence_dir) as workspace,
            package(
                repository,
                evidence_dir,
                slot.session,
                workload_uuid=slot.uuid,
                private_seed=base_seed(slot.session.case, key),
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
                evidence_dir,
                device="cuda:0",
                start=time.monotonic(),
                deadline=deadline,
                instrument=partial(
                    instrument_holdout,
                    shift=slot.shift,
                    oracle_cpu_allowance_seconds=ORACLE_SECONDS_PER_SESSION,
                    holdout_commitment_sha256=commitment_sha,
                ),
            )
        result.update(
            **workspace,
            candidate_sha256=candidate,
            reference_sha256=reference,
            feature_shift=slot.shift,
        )
    finally:
        (evidence_dir / PRIVATE_FILE).unlink(missing_ok=True)
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
    return result


def _owned_online(
    repository: Path,
    destination: Path,
    spec: dict[str, Any],
    freeze_sha: str,
) -> None:
    locks = configured_gpu_locks()
    owner = verify_clock_owner(
        locks.directory, os.environ[ENV_CLOCK_OWNER], (resolve_gpu_device(),)
    )
    if not owner.container_supervised:
        raise ValueError(
            "LayerNorm holdout requires a container-supervised owner"
        )
    destination.mkdir(exist_ok=False)
    atomic_write_json_value(
        destination / "online-owner.json", owner.model_dump(mode="json")
    )
    key, commitment_sha = _commit_holdout(destination, spec, freeze_sha)
    online_dir = destination / "online"
    online_dir.mkdir(exist_ok=False)
    started = time.monotonic()
    stop = None
    completed = 0
    try:
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
            slot_dir = online_dir / slot.name
            result = _run_slot(
                repository,
                slot_dir,
                slot,
                remaining,
                key=key,
                commitment_sha=commitment_sha,
            )
            atomic_write_json_value(
                slot_dir / "evidence" / "session.json", result
            )
            completed += 1
            if result["stop_reason"]:
                stop = result["stop_reason"]
                break
    finally:
        state = (
            "completed"
            if completed == len(spec["roster"]) and stop is None
            else "INCOMPLETE"
        )
        atomic_write_json_value(
            destination / "online-summary.json",
            {
                "state": state,
                "freeze_sha256": freeze_sha,
                "commitment_sha256": commitment_sha,
                "roster_sha256": spec["roster_sha256"],
                "completed_sessions": completed,
                "stop_reason": stop,
                "wall_seconds": time.monotonic() - started,
                "performance_authority": False,
            },
        )
        _write_exclusive(destination / CLOSED_KEY_FILE, key)


def _verify_only(repository: Path, bundle: Path, expected_sha: str) -> None:
    if Path("/dev/kfd").exists() or any(Path("/dev/dri").glob("renderD*")):
        raise ValueError(
            "LayerNorm holdout readiness check requires a device-free container"
        )
    spec = verify_bundle(repository, bundle, expected_sha)
    if (bundle / ACTIVITY_DIRECTORY).exists():
        raise FileExistsError(bundle / ACTIVITY_DIRECTORY)
    print(
        json.dumps(
            {
                "verified": True,
                "freeze_sha256": expected_sha,
                "roster_slots": len(spec["roster"]),
                "holdout_opened": False,
            },
            sort_keys=True,
        )
    )


def _open_holdout(repository: Path, bundle: Path, expected_sha: str) -> None:
    spec = verify_bundle(repository, bundle, expected_sha)
    if (
        os.environ.get(ENV_SOL_EXECBENCH_SANDBOXED) != "1"
        or not configured_gpu_locks().host_managed
    ):
        raise ValueError(
            "LayerNorm holdout validation requires the hardened native owner"
        )
    require_empty_scratch()
    if sha256_file(Path("/proc/1/exe")) != spec["docker_init"]["sha256"]:
        raise ValueError("LayerNorm holdout Docker init differs from freeze")
    with native_gpu_execution(manage_clocks=True):
        physical = resolve_gpu_device()
        if (physical.uuid, physical.bdf) != (
            spec["device"]["uuid"],
            spec["device"]["bdf"],
        ):
            raise ValueError("LayerNorm holdout targeted another GPU")
        dev = torch.cuda.get_device_properties(0)
        if (
            torch.cuda.device_count() != 1
            or dev.gcnArchName.split(":")[0] != "gfx1200"
        ):
            raise ValueError("LayerNorm holdout runtime does not match gfx1200")
        protect_observation_process()
        _owned_online(
            repository,
            bundle / ACTIVITY_DIRECTORY,
            spec,
            expected_sha,
        )


def main() -> None:
    """Verify frozen readiness or run the explicitly authorized h0 session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--open-holdout", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        _verify_only(args.repository, args.bundle, args.expected_sha)
    else:
        _open_holdout(args.repository, args.bundle, args.expected_sha)


if __name__ == "__main__":
    main()
