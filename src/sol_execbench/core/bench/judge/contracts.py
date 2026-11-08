"""Frozen development cases and bounded judge diagnostic vocabulary."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import load_json_file, load_jsonl_file
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_file


class JudgeSpec(StrEnum):
    """Frozen research protocol selector, not an official scoring contract."""

    M0 = "M0-E0-20261107-r1"


MAX_INPUT_ELEMENTS = 4_194_304
MAX_OUTPUT_ELEMENTS = 262_144
MAX_DOT_PRODUCTS = 33_554_432
SMALL_VALUE = 2.0**-20
FLUSH_THRESHOLD = 2.0**-12
DEVELOPMENT_SEEDS = (1701, 1702)
REPETITIONS = 3


class JudgeTask(StrEnum):
    """Frozen operation families."""

    GEMM = "gemm"
    SUM = "sum"
    LAYERNORM = "layernorm"


class JudgeStrategy(StrEnum):
    """The four preregistered interventions."""

    REFERENCE = "p0"
    SEMANTIC = "p1"
    NUMERICAL = "p2"
    ALTERNATIVE = "p3"


class JudgeInstance(StrEnum):
    """Exposure roles; holdout materialization is deliberately unavailable."""

    DEVELOPMENT_0 = "d0"
    DEVELOPMENT_1 = "d1"
    HOLDOUT = "h0"


class JudgeFamily(StrEnum):
    """Input families fixed before candidate evaluation."""

    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


TASK_PROBLEMS = {
    JudgeTask.GEMM: "l1n2_standard_matmul",
    JudgeTask.SUM: "l1n47_sum_reduction",
    JudgeTask.LAYERNORM: "l1n40_layernorm",
}
TASK_FAMILIES = {
    JudgeTask.GEMM: tuple(
        JudgeFamily(value) for value in ("G0", "G1", "G2", "G3")
    ),
    JudgeTask.SUM: tuple(
        JudgeFamily(value) for value in ("R0", "R1", "R2", "R3")
    ),
    JudgeTask.LAYERNORM: tuple(
        JudgeFamily(value) for value in ("L0", "L1", "L2", "L3")
    ),
}
WORKLOAD_INDICES = (0, 1, 2)
DEVELOPMENT_INSTANCES = (
    JudgeInstance.DEVELOPMENT_0,
    JudgeInstance.DEVELOPMENT_1,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class JudgeCase:
    """A logical case, without inputs, secrets, or a candidate result."""

    task: JudgeTask
    workload_index: int
    family: JudgeFamily
    instance: JudgeInstance

    def __post_init__(self) -> None:
        """Reject invalid selections before input allocation."""
        if not isinstance(self.task, JudgeTask):
            raise TypeError("task must be JudgeTask")
        if self.family not in TASK_FAMILIES[self.task]:
            raise ValueError("family does not belong to task")
        if self.workload_index not in WORKLOAD_INDICES:
            raise ValueError("workload index is outside the frozen denominator")
        if not isinstance(self.instance, JudgeInstance):
            raise TypeError("instance must be JudgeInstance")

    @property
    def logical_id(self) -> str:
        """Return the frozen identity, independent of repetition."""
        return (
            f"{JudgeSpec.M0}/{self.task}/{self.workload_uuid}/"
            f"{self.family}/{self.instance}"
        )

    @property
    def workload_uuid(self) -> str:
        """Return the original AKA workload identifier."""
        return f"aka-{TASK_PROBLEMS[self.task]}-w{self.workload_index}"

    def require_development(self) -> int:
        """Reject holdout before loading or generating any input."""
        if self.instance not in DEVELOPMENT_INSTANCES:
            raise ValueError("holdout generation is unavailable before E2")
        return DEVELOPMENT_SEEDS[DEVELOPMENT_INSTANCES.index(self.instance)]


def development_cases() -> tuple[JudgeCase, ...]:
    """Enumerate only the 72 development inputs, without materializing them."""
    return tuple(
        JudgeCase(
            task=task, workload_index=index, family=family, instance=instance
        )
        for task in JudgeTask
        for index in WORKLOAD_INDICES
        for family in TASK_FAMILIES[task]
        for instance in DEVELOPMENT_INSTANCES
    )


def load_case_problem(
    root: Path, case: JudgeCase
) -> tuple[Definition, Workload]:
    """Read one existing problem; never rewrite the published corpus."""
    case.require_development()
    directory = (
        root / "problems" / "AMD_AKA" / "torch2hip" / TASK_PROBLEMS[case.task]
    )
    for name, digest in zip(
        ("definition.json", "workload.jsonl"),
        FROZEN_SOURCES[case.task],
        strict=True,
    ):
        if sha256_file(directory / name) != digest:
            raise ValueError("frozen judge source drift")
    definition = load_json_file(Definition, directory / "definition.json")
    workloads = load_jsonl_file(Workload, directory / "workload.jsonl")
    selected = [item for item in workloads if item.uuid == case.workload_uuid]
    if len(selected) != 1:
        raise ValueError("frozen judge workload is missing or duplicated")
    return definition, selected[0]


FROZEN_SOURCES = {
    JudgeTask.GEMM: (
        "32983314f87a12778e973d2557d33ae06c8fabf27459a8b7d4edc4cda17e4742",
        "1884b69eb45cbc18a53d3bc746fbe0ccaaa5102a8964daab189f22ee973a0849",
    ),
    JudgeTask.SUM: (
        "69e4a427d37168da498c1cd9b81b63df9a1637880b2c71ba79aeea9135055136",
        "8b1bd83967d61a7e6a5a15be0ec58703e2beefef96b709939d9276c231623ef6",
    ),
    JudgeTask.LAYERNORM: (
        "949f4ca1d7c6214a02427209bdd68786a8dc9eb724e52bd9e0ead57bbcdca557",
        "18096048cf9462ed546db2c3b895eae85c7488943f40a530a485dd610b93aafa",
    ),
}
