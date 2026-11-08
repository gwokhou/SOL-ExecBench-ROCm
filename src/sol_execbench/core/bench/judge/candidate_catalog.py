"""Packaged candidate source identities, independent of oracle construction."""

from collections.abc import Callable
from importlib import resources

import torch

from sol_execbench.core.bench.judge.candidates import (
    gemm_p0,
    gemm_p1,
    gemm_p2,
    gemm_p3,
    layernorm_p0,
    layernorm_p1,
    layernorm_p2,
    layernorm_p3,
    sum_p0,
    sum_p1,
    sum_p2,
    sum_p3,
)
from sol_execbench.core.bench.judge.contracts import JudgeStrategy, JudgeTask
from sol_execbench.core.data.solution_instance import Solution
from sol_execbench.core.data.solution_models import (
    BuildSpec,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)

_PACKAGE = "sol_execbench.core.bench.judge.candidates"
_ENTRY_POINT = "candidate.py::run"


def candidate_source(task: JudgeTask, strategy: JudgeStrategy) -> str:
    """Load the actual distributed source, without importing candidate code."""
    return (
        resources.files(_PACKAGE)
        .joinpath(f"{task}_{strategy}.py")
        .read_text(encoding="utf-8")
    )


def candidate_solution(
    task: JudgeTask,
    strategy: JudgeStrategy,
    definition_name: str,
) -> Solution:
    """Package a candidate for the incumbent evaluator and static review."""
    return Solution(
        name=f"judge-{task}-{strategy}",
        definition=definition_name,
        author="M0-E0-preregistered-control",
        spec=BuildSpec(
            languages=[SupportedLanguages.PYTORCH],
            target_hardware=[SupportedHardware.GFX1200],
            entry_point=_ENTRY_POINT,
            destination_passing_style=False,
        ),
        sources=[
            SourceFile(
                path="candidate.py", content=candidate_source(task, strategy)
            )
        ],
    )


def candidate_callable(
    task: JudgeTask,
    strategy: JudgeStrategy,
) -> Callable[..., torch.Tensor]:
    """Load only owned controls for offline CPU diagnostics, never user code."""
    return _CALLABLES[(task, strategy)]


_CALLABLES = {
    (JudgeTask.GEMM, JudgeStrategy.REFERENCE): gemm_p0.run,
    (JudgeTask.GEMM, JudgeStrategy.SEMANTIC): gemm_p1.run,
    (JudgeTask.GEMM, JudgeStrategy.NUMERICAL): gemm_p2.run,
    (JudgeTask.GEMM, JudgeStrategy.ALTERNATIVE): gemm_p3.run,
    (JudgeTask.SUM, JudgeStrategy.REFERENCE): sum_p0.run,
    (JudgeTask.SUM, JudgeStrategy.SEMANTIC): sum_p1.run,
    (JudgeTask.SUM, JudgeStrategy.NUMERICAL): sum_p2.run,
    (JudgeTask.SUM, JudgeStrategy.ALTERNATIVE): sum_p3.run,
    (JudgeTask.LAYERNORM, JudgeStrategy.REFERENCE): layernorm_p0.run,
    (JudgeTask.LAYERNORM, JudgeStrategy.SEMANTIC): layernorm_p1.run,
    (JudgeTask.LAYERNORM, JudgeStrategy.NUMERICAL): layernorm_p2.run,
    (JudgeTask.LAYERNORM, JudgeStrategy.ALTERNATIVE): layernorm_p3.run,
}
