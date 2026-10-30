"""Semantic runtime drift is explicit and hashing never queries GPU devices."""

import pytest
import torch

from solar.verification import execution
from solar.verification.contracts import VerificationPolicy
from solar.verification.errors import VerificationError
from solar.verification.policy import MAX_VERIFICATION_CASES, verification_cases


@pytest.mark.parametrize(
    "field",
    [
        "python_implementation",
        "python_version",
        "torch_version",
        "semantic_implementation_sha256",
    ],
)
def test_changed_or_missing_semantic_runtime_is_rejected(field):
    recorded = execution.execution_identity("cpu")
    recorded[field] = "changed"
    with pytest.raises(VerificationError, match=field):
        execution.require_current_semantic_runtime(recorded)
    del recorded[field]
    with pytest.raises(VerificationError, match=field):
        execution.require_current_semantic_runtime(recorded)


def test_runtime_identity_does_not_enumerate_gpu(monkeypatch):
    def forbidden():
        pytest.fail("CPU identity must not query GPU availability")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    observed = execution.execution_identity("cpu")
    assert observed["backend"] == "cpu"
    assert len(observed["semantic_implementation_sha256"]) == 64


@pytest.fixture
def semantic_sources(tmp_path, monkeypatch):
    root = tmp_path / "solar"
    (root / "verification").mkdir(parents=True)
    for name in execution.SEMANTIC_SOURCE_FILES:
        (root / name).write_text("x = 10 ** 30\n")
    path = root / "verification" / "execution.py"
    path.write_text("# fixture interpreter\n")
    monkeypatch.setattr(execution, "__file__", str(path))
    return root


def test_interpreter_changes_invalidate_identity_without_a_schema_edit(
    semantic_sources,
):
    before = execution.semantic_runtime_identity()
    (semantic_sources / "storage.py").write_text("x = 10 ** 31\n")
    after = execution.semantic_runtime_identity()
    assert (
        before["semantic_implementation_sha256"]
        != after["semantic_implementation_sha256"]
    )


@pytest.mark.parametrize("budget", ["files", "bytes"])
def test_semantic_source_work_is_bounded(semantic_sources, monkeypatch, budget):
    name = (
        "MAX_SEMANTIC_SOURCE_FILES"
        if budget == "files"
        else "MAX_SEMANTIC_SOURCE_BYTES"
    )
    monkeypatch.setattr(execution, name, 2)
    with pytest.raises(VerificationError, match="budget"):
        execution.semantic_runtime_identity()


def test_small_source_with_large_nominal_numeric_value_is_accepted(
    semantic_sources,
):
    assert execution.semantic_runtime_identity()[
        "semantic_implementation_sha256"
    ]


def test_case_product_is_bounded_before_enumeration():
    policy = VerificationPolicy(
        atol=0, rtol=0, seeds=range(MAX_VERIFICATION_CASES * 1000000000)
    )
    with pytest.raises(VerificationError, match="case budget"):
        verification_cases({}, policy)


def test_large_seed_values_do_not_exceed_a_small_case_budget():
    cases = verification_cases(
        {},
        VerificationPolicy(
            atol=0, rtol=0, seeds=(10**30, 10**30 + 1, 10**30 + 2)
        ),
    )
    assert len(cases) == 9
