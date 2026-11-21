"""The opt-in LayerNorm reference and its trusted timing path on CPU."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts.internal.solar.e2_contracts import Activity, Mode, Session
from scripts.internal.solar.e2_layernorm_reference import run
from scripts.internal.solar.e2_staging import LayerNormReference, package
from sol_execbench.core.bench.eval_runtime import load_reference_function
from sol_execbench.core.bench.judge.candidate_catalog import candidate_source
from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeFamily,
    JudgeInstance,
    JudgeStrategy,
    JudgeTask,
)
from sol_execbench.core.bench.judge.diagnostics import stronger_check
from sol_execbench.core.bench.judge.inputs import generate_seeded_inputs
from sol_execbench.core.bench.judge.interval_oracle import layernorm_oracle
from sol_execbench.core.bench.reference_service import (
    ReferenceRequestError,
    ReferenceService,
)

ROOT = Path(__file__).resolve().parents[2]
CASE = JudgeCase(
    task=JudgeTask.LAYERNORM,
    workload_index=0,
    family=JudgeFamily.L1,
    instance=JudgeInstance.DEVELOPMENT_0,
)
SESSION = Session(
    case=CASE,
    strategy=JudgeStrategy.REFERENCE,
    repeat=0,
    mode=Mode.DIAGNOSTIC,
    activity=Activity.CORRECTNESS,
)
WORKLOAD = "layernorm-reference-cpu"
RESEARCH = LayerNormReference.CENTERED_FP64_V1


@pytest.mark.parametrize(
    "family",
    (JudgeFamily.L0, JudgeFamily.L1, JudgeFamily.L2, JudgeFamily.L3),
    ids=("offset", "near-constant", "epsilon", "affine"),
)
def test_reference_agrees_with_the_independent_interval(family):
    inputs = generate_seeded_inputs(
        CASE.task, family, {"B": 6, "N": 128}, CASE.require_development()
    )
    oracle, _, _ = layernorm_oracle(inputs.values)
    actual = run(*inputs.values)
    assert actual.dtype == torch.float32
    assert actual.shape == oracle.shape
    assert stronger_check(replace(CASE, family=family), inputs, actual, oracle)


@pytest.mark.parametrize("strategy", list(JudgeStrategy), ids=str)
def test_staged_variant_updates_only_the_reference_and_equivalent_control(
    tmp_path, strategy
):
    session = replace(SESSION, strategy=strategy)
    with (
        package(
            ROOT,
            tmp_path / "original",
            session,
            workload_uuid=WORKLOAD,
            private_seed=1701,
        ) as original,
        package(
            ROOT,
            tmp_path / "centered",
            session,
            workload_uuid=WORKLOAD,
            private_seed=1701,
            layernorm_reference=RESEARCH,
        ) as centered,
    ):
        incumbent_source = candidate_source(CASE.task, strategy)
        assert original.solution.sources[0].content == incumbent_source
        assert original.workloads == centered.workloads
        assert original.config == centered.config
        assert original.definition.reference != centered.definition.reference
        assert original.definition.model_dump(exclude={"reference"}) == (
            centered.definition.model_dump(exclude={"reference"})
        )
        staged = centered.solution.sources[0].content
        assert (
            centered.output_dir / "candidate" / "candidate.py"
        ).read_text() == staged
        if strategy is JudgeStrategy.REFERENCE:
            assert staged != incumbent_source
            assert centered.definition.reference.startswith(staged)
            assert centered.solution.hash() != original.solution.hash()
        else:
            assert staged == incumbent_source


def test_reference_variant_rejects_non_layernorm(tmp_path):
    gemm = replace(CASE, task=JudgeTask.GEMM, family=JudgeFamily.G0)
    with pytest.raises(ValueError, match="requires a LayerNorm session"):
        package(
            ROOT,
            tmp_path,
            replace(SESSION, case=gemm),
            workload_uuid=WORKLOAD,
            private_seed=1701,
            layernorm_reference=RESEARCH,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "bias_shift",
    [0.0, 1.8848630134016275e-5],
    ids=("legal", "outside-tolerance"),
)
def test_trusted_timing_checks_the_new_reference(tmp_path, bias_shift):
    with package(
        ROOT,
        tmp_path,
        SESSION,
        workload_uuid=WORKLOAD,
        private_seed=1701,
        layernorm_reference=RESEARCH,
    ) as staged:
        service = ReferenceService(
            tmp_path, device="cpu", input_nonce="a1" * 32
        )
        _, timing, _, _ = service.handle(
            {
                "operation": "timing_iteration",
                "row_index": 0,
                "round_index": 9,
                "workload_uuid": WORKLOAD,
                "trial_index": 0,
                "iteration_index": 0,
            }
        )
        assert timing.outputs == []
        _, candidate = load_reference_function(
            (staged.output_dir / "candidate" / "candidate.py").read_text()
        )
        with torch.inference_mode():
            oracle, _, _ = layernorm_oracle(
                (*timing.inputs[:3], torch.tensor(timing.inputs[3]))
            )
            actual = candidate(*timing.inputs)
            assert torch.equal(actual, oracle)
            if bias_shift:
                with pytest.raises(
                    ReferenceRequestError, match="timed output differs"
                ):
                    service.validate_timing_outputs([actual + bias_shift])
            else:
                service.validate_timing_outputs([actual])
    assert not torch.cuda.is_initialized()
