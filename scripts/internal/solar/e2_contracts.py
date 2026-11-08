"""Frozen E0 experiment schedule; no input or holdout materialization."""

import hashlib
import hmac
from dataclasses import dataclass
from enum import StrEnum

from sol_execbench.core.bench.judge.contracts import (
    REPETITIONS,
    TASK_FAMILIES,
    WORKLOAD_INDICES,
    JudgeCase,
    JudgeInstance,
    JudgeStrategy,
    JudgeTask,
)

PRIVATE_FILE = "judge-private.json"
OBSERVATION_OPERATION = "judge_correctness_observation_v1"
MAX_OBSERVATION_BYTES = 524_288
ORACLE_TIMEOUT_SECONDS = 120
CORRECTNESS_TIMEOUT_SECONDS = 120
COST_TIMEOUT_SECONDS = 300
GPU_BUDGET_SECONDS = 21_600
ORACLE_BUDGET_SECONDS = 43_200
EVIDENCE_BUDGET_BYTES = 20 * 1024**3
MAX_SESSION_OBSERVATIONS = 190


class Mode(StrEnum):
    """Research observation mode; never a production scoring selector."""

    INCUMBENT = "incumbent"
    DIAGNOSTIC = "diagnostic"


class Activity(StrEnum):
    """The two preregistered activity denominators."""

    CORRECTNESS = "correctness"
    COST = "cost"


@dataclass(frozen=True, slots=True, kw_only=True)
class Session:
    """One independent process repetition of a fixed candidate-case unit."""

    case: JudgeCase
    strategy: JudgeStrategy
    repeat: int
    mode: Mode
    activity: Activity

    @property
    def identity(self) -> str:
        """Return an identifier containing no generated input information."""
        return (
            f"{self.activity}/{self.case.logical_id}/{self.strategy}/"
            f"r{self.repeat}/{self.mode}"
        )

    def metadata(self) -> dict[str, object]:
        """Retain all denominator coordinates, including not-reached sessions."""
        return {
            "session_id": self.identity,
            "case_id": self.case.logical_id,
            "task": self.case.task.value,
            "workload_index": self.case.workload_index,
            "original_workload_uuid": self.case.workload_uuid,
            "family": self.case.family.value,
            "instance": self.case.instance.value,
            "strategy": self.strategy.value,
            "repeat": self.repeat,
            "mode": self.mode.value,
            "activity": self.activity.value,
        }


def schedule() -> tuple[Session, ...]:
    """Enumerate 1296 correctness and 144 cost sessions without selection."""
    correctness = [
        Session(
            case=case,
            strategy=strategy,
            repeat=repeat,
            mode=Mode.DIAGNOSTIC,
            activity=Activity.CORRECTNESS,
        )
        for task in JudgeTask
        for index in WORKLOAD_INDICES
        for family in TASK_FAMILIES[task]
        for instance in JudgeInstance
        for case in [
            JudgeCase(
                task=task,
                workload_index=index,
                family=family,
                instance=instance,
            )
        ]
        for strategy in JudgeStrategy
        for repeat in range(REPETITIONS)
    ]
    costs = [
        Session(
            case=case,
            strategy=strategy,
            repeat=repeat,
            mode=mode,
            activity=Activity.COST,
        )
        for task in JudgeTask
        for family in TASK_FAMILIES[task]
        for case in [
            JudgeCase(
                task=task,
                workload_index=0,
                family=family,
                instance=JudgeInstance.DEVELOPMENT_0,
            )
        ]
        for strategy in (JudgeStrategy.REFERENCE, JudgeStrategy.ALTERNATIVE)
        for repeat in range(REPETITIONS)
        for mode in (tuple(Mode) if repeat % 2 == 0 else tuple(reversed(Mode)))
    ]
    return tuple(correctness + costs)


def base_seed(case: JudgeCase, key: bytes) -> int:
    """Apply E0's exact HMAC byte order; the key never enters candidate state."""
    if case.instance is not JudgeInstance.HOLDOUT:
        return case.require_development()
    if len(key) != 32:
        raise ValueError("holdout key must contain exactly 32 bytes")
    digest = hmac.new(
        key, case.logical_id.encode("utf-8"), hashlib.sha256
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF
