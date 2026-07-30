from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from solar.ir.contracts import IRKind
from solar.ir.registry import ir_lifecycle
from solar.schema_versions import ATEN_IR_SCHEMA_VERSION
from solar.verification import numerics, verify as verification
from solar.verification.verify import VerificationError

REFERENCE_SOURCE = """
import torch

def reference(value):
    return value

def make_inputs(parameters, device):
    seed = int(parameters['seed'])
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return [torch.randn(2, 2, generator=generator, device=device)]
"""


def _identity_graph() -> dict:
    return {
        "schema_version": ATEN_IR_SCHEMA_VERSION,
        "ir_kind": "aten",
        "layers": {
            "start": {
                "type": "start",
                "semantic_op": {
                    "kind": "input",
                    "target": "input",
                    "arguments": [],
                    "kwargs": {},
                },
                "tensor_names": {"inputs": [], "outputs": ["input"]},
                "tensor_shapes": {"inputs": [], "outputs": [[2, 2]]},
                "tensor_dtypes": {"inputs": [], "outputs": ["torch.float32"]},
            },
            "identity": {
                "type": "identity",
                "semantic_op": {
                    "kind": "aten",
                    "target": "identity",
                    "overload": "default",
                    "arguments": [{"tensor": 0}],
                    "kwargs": {},
                    "effects": {
                        "mutates": [],
                        "aliases": [{"output": 0, "input": 0}],
                        "atomic": False,
                        "opaque_library_call": False,
                    },
                },
                "tensor_names": {"inputs": ["input"], "outputs": ["output"]},
                "tensor_shapes": {"inputs": [[2, 2]], "outputs": [[2, 2]]},
                "tensor_dtypes": {
                    "inputs": ["torch.float32"],
                    "outputs": ["torch.float32"],
                },
            },
        },
        "outputs": ["output"],
    }


@pytest.fixture
def verification_files(tmp_path: Path):
    reference = tmp_path / "reference.py"
    reference.write_text(REFERENCE_SOURCE)
    graph = tmp_path / "graph.yaml"
    graph.write_text(yaml.safe_dump(_identity_graph(), sort_keys=False))
    return reference, graph


def test_create_and_replay_source_verification_artifact(
    tmp_path: Path,
    verification_files,
) -> None:
    reference, graph = verification_files
    output = tmp_path / "verification.yaml"

    artifact = verification.create_verification_artifact(
        reference_path=reference,
        reference_entry_point="reference",
        input_factory_name="make_inputs",
        graph_path=graph,
        workload_name="identity",
        workload_parameters={},
        output_path=output,
        policy=verification.VerificationPolicy(atol=0.0, rtol=0.0),
        lifecycle=ir_lifecycle(IRKind.ATEN),
    )

    assert yaml.safe_load(output.read_text()) == artifact
    assert len(artifact["predicate"]["cases"]) == 9
    verification.replay_verification_artifact(
        artifact,
        reference_path=reference,
        graph_path=graph,
        workload_name="identity",
        workload_parameters={},
        required_tolerance=verification.TolerancePolicy(
            atol=0.0,
            rtol=0.0,
        ),
        lifecycle=ir_lifecycle(IRKind.ATEN),
    )


def test_callable_verification_writes_hash_bound_attestation(
    tmp_path: Path,
    verification_files,
) -> None:
    _, graph = verification_files
    output = tmp_path / "callable.yaml"

    artifact = verification.verify_callable_conversion(
        reference=lambda value: value,
        input_factory=lambda seed: [torch.full((2, 2), float(seed))],
        reference_name="definition.py#reference",
        reference_sha256="a" * 64,
        graph_path=graph,
        output_path=output,
        policy=verification.VerificationPolicy(atol=0.0, rtol=0.0),
        lifecycle=ir_lifecycle(IRKind.ATEN),
    )

    assert artifact["subject"][0]["digest"]["sha256"] == "a" * 64
    assert len(artifact["predicate"]["results"]) == 9
    assert yaml.safe_load(output.read_text()) == artifact


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (
            verification.create_verification_artifact,
            {
                "policy": verification.VerificationPolicy(
                    atol=0.0,
                    rtol=0.0,
                    seeds=(1, 1, 2),
                ),
            },
            "at least three seeds",
        ),
        (
            verification.create_verification_artifact,
            {
                "policy": verification.VerificationPolicy(
                    atol=0.0,
                    rtol=0.0,
                    patterns=("random", "zeros"),
                ),
            },
            "boundary patterns",
        ),
        (
            verification.verify_callable_conversion,
            {"reference_sha256": "BAD"},
            "lowercase SHA-256",
        ),
        (
            verification.verify_callable_conversion,
            {
                "policy": verification.VerificationPolicy(
                    atol=0.0,
                    rtol=0.0,
                    seeds=(1, 1, 2),
                ),
            },
            "at least three seeds",
        ),
        (
            verification.verify_callable_conversion,
            {
                "policy": verification.VerificationPolicy(
                    atol=0.0,
                    rtol=0.0,
                    patterns=("random", "zeros"),
                ),
            },
            "boundary patterns",
        ),
    ],
)
def test_artifact_creation_rejects_weak_case_sets(
    tmp_path: Path,
    verification_files,
    function,
    kwargs,
    message,
) -> None:
    reference, graph = verification_files
    common = {
        "graph_path": graph,
        "output_path": tmp_path / "out.yaml",
        "policy": verification.VerificationPolicy(atol=0.0, rtol=0.0),
        "lifecycle": ir_lifecycle(IRKind.ATEN),
    }
    if function is verification.create_verification_artifact:
        common.update(
            reference_path=reference,
            reference_entry_point="reference",
            input_factory_name="make_inputs",
            workload_name="identity",
            workload_parameters={},
        )
    else:
        common.update(
            reference=lambda value: value,
            input_factory=lambda seed: [torch.ones(2, 2)],
            reference_name="reference",
            reference_sha256="a" * 64,
        )
    common.update(kwargs)
    with pytest.raises(VerificationError, match=message):
        function(**common)


def test_run_cases_validates_source_input_indices(verification_files) -> None:
    _, graph_path = verification_files
    graph = yaml.safe_load(graph_path.read_text())
    cases = [{"seed": 1, "pattern": "random", "parameters": {}}]
    common: dict[str, Any] = {
        "reference": lambda *values: values[0],
        "input_factory": lambda parameters, device: [torch.ones(2, 2), 4],
        "lifecycle": ir_lifecycle(IRKind.ATEN),
        "cases": cases,
        "tolerance": verification.TolerancePolicy(atol=0.0, rtol=0.0),
        "device": "cpu",
        "check_shapes": True,
    }
    graph["source_input_indices"] = [9]
    with pytest.raises(VerificationError, match="invalid source_input_indices"):
        verification._run_cases(graph=graph, **common)
    graph["source_input_indices"] = [1]
    with pytest.raises(VerificationError, match="must select tensor"):
        verification._run_cases(graph=graph, **common)


def test_pattern_inputs_cover_float_bool_integer_and_unknown() -> None:
    source = (
        torch.ones(3),
        torch.tensor([True, False, True]),
        torch.tensor([1, 2, 3]),
        "value",
    )
    assert numerics.pattern_inputs(source, "random") == source
    zeros = numerics.pattern_inputs(source, "zeros")
    assert torch.equal(zeros[0], torch.zeros(3))
    boundary = numerics.pattern_inputs(source, "boundary")
    assert boundary[0].tolist() == [-1.0, 0.0, 1.0]
    assert boundary[1].tolist() == [False, True, False]
    assert boundary[2].tolist() == [0, 0, 0]
    assert boundary[3] == "value"
    protected = numerics.pattern_inputs(
        source,
        "boundary",
        preserved_input_indices=(0, 2),
    )
    assert protected[0] is source[0]
    assert protected[2] is source[2]
    assert protected[1].tolist() == [False, True, False]
    with pytest.raises(
        VerificationError,
        match="unknown verification input pattern",
    ):
        numerics.pattern_inputs(source, "missing")


def test_assert_close_reports_success_for_supported_output_shapes() -> None:
    assert numerics.assert_close(torch.ones(2), torch.ones(2), 0, 0) == {
        "max_abs_error": 0.0,
        "matched_ratio": 1.0,
    }
    assert (
        numerics.assert_close(
            torch.tensor([-torch.inf]),
            torch.tensor([-torch.inf]),
            0,
            0,
            allow_negative_inf=True,
        )["matched_ratio"]
        == 1.0
    )
    matching_nan = numerics.assert_close(
        torch.tensor([torch.nan]),
        torch.tensor([torch.nan]),
        0,
        0,
        allow_matching_nan=True,
    )
    assert matching_nan["matching_nan_count"] == 1.0
    assert (
        numerics.assert_close((torch.ones(1),), (torch.ones(1),), 0, 0)[
            "max_abs_error"
        ]
        == 0.0
    )
    assert (
        numerics.assert_close(
            {"x": torch.ones(1)},
            {"x": torch.ones(1)},
            0,
            0,
        )["max_abs_error"]
        == 0.0
    )
    assert numerics.assert_close("x", "x", 0, 0)["max_abs_error"] == 0.0


@pytest.mark.parametrize(
    ("actual", "expected", "atol", "max_error_cap", "message"),
    (
        pytest.param(
            torch.ones(2),
            torch.full((2,), 2.0),
            0,
            None,
            "numerical mismatch",
            id="outside-tolerance",
        ),
        pytest.param(
            torch.tensor([1.1]),
            torch.tensor([1.0]),
            1,
            0.01,
            "exceeds cap",
            id="above-error-cap",
        ),
    ),
)
def test_assert_close_rejects_float_values_outside_policy(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    max_error_cap: float | None,
    message: str,
) -> None:
    with pytest.raises(VerificationError, match=message):
        numerics.assert_close(
            actual,
            expected,
            atol,
            0,
            max_error_cap=max_error_cap,
        )


@pytest.mark.parametrize(
    ("actual", "expected", "message"),
    (
        pytest.param(
            torch.ones(2),
            torch.ones(3),
            "shape mismatch",
            id="tensor-shape",
        ),
        pytest.param(
            torch.ones(2),
            torch.ones(2, dtype=torch.float16),
            "dtype mismatch",
            id="tensor-dtype",
        ),
        pytest.param(
            torch.tensor([1]),
            torch.tensor([2]),
            "integer/bool tensor values differ",
            id="integer-values",
        ),
        pytest.param(
            torch.tensor([torch.nan]),
            torch.tensor([torch.nan]),
            "non-finite",
            id="non-finite-values",
        ),
        pytest.param(
            torch.zeros(2),
            torch.ones(2),
            "all-zero output",
            id="all-zero-output",
        ),
        pytest.param(
            (torch.ones(1),),
            (torch.ones(1), torch.ones(1)),
            "arity mismatch",
            id="sequence-arity",
        ),
        pytest.param(
            {"x": 1},
            {"y": 1},
            "mapping keys differ",
            id="mapping-keys",
        ),
        pytest.param(
            "x",
            "y",
            "non-tensor output mismatch",
            id="scalar-value",
        ),
    ),
)
def test_assert_close_rejects_invalid_values(
    actual: Any,
    expected: Any,
    message: str,
) -> None:
    with pytest.raises(VerificationError, match=message):
        numerics.assert_close(actual, expected, 0, 0)


def test_pure_contraction_roundoff_requires_both_results_within_gamma_bound() -> (
    None
):
    graph = {
        "layers": {
            "contract": {
                "type": "matmul",
                "semantic_op": {
                    "kind": "einsum",
                    "equation": "MK,KN->MN",
                },
            },
        },
    }
    left = torch.ones(2, 16)
    right = torch.ones(16, 1)
    expected = torch.matmul(left, right)

    assert verification._einsum_roundoff_equivalent(
        graph,
        (left, right),
        expected + 1e-5,
        expected,
    )
    assert not verification._einsum_roundoff_equivalent(
        graph,
        (left, right),
        expected + 0.1,
        expected,
    )


@pytest.fixture
def source_artifact(tmp_path: Path, verification_files):
    reference, graph = verification_files
    return (
        verification.create_verification_artifact(
            reference_path=reference,
            reference_entry_point="reference",
            input_factory_name="make_inputs",
            graph_path=graph,
            workload_name="identity",
            workload_parameters={},
            output_path=tmp_path / "verification.yaml",
            policy=verification.VerificationPolicy(atol=0.0, rtol=0.0),
            lifecycle=ir_lifecycle(IRKind.ATEN),
        ),
        reference,
        graph,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(_type="bad"), "in-toto Statement"),
        (lambda value: value.update(predicateType="bad"), "predicate type"),
        (
            lambda value: value["predicate"].update(status="failed"),
            "not a trusted passing result",
        ),
        (
            lambda value: value["predicate"]["workload"].update(name="other"),
            "workload name mismatch",
        ),
        (
            lambda value: value["predicate"]["workload"].update(
                parameters_sha256="0" * 64,
            ),
            "workload parameters mismatch",
        ),
        (
            lambda value: value["predicate"]["tolerance"].update(atol=1.0),
            "tolerance is weaker",
        ),
        (
            lambda value: value["predicate"].update(results=[]),
            "lacks the required cases",
        ),
        (
            lambda value: value["predicate"]["cases"].__setitem__(
                slice(None),
                value["predicate"]["cases"][:2],
            ),
            "lacks the required cases",
        ),
        (
            lambda value: value["predicate"]["execution"].update(
                device_type="tpu",
            ),
            "no supported replay device",
        ),
        (
            lambda value: value["predicate"]["execution"].update(backend="tpu"),
            "no execution backend identity",
        ),
    ],
    ids=(
        "statement-type",
        "predicate-type",
        "predicate-status",
        "workload-name",
        "workload-parameters",
        "tolerance",
        "empty-results",
        "missing-required-case",
        "unsupported-device",
        "unsupported-backend",
    ),
)
def test_replay_rejects_untrusted_artifact_mutations(
    source_artifact,
    mutate,
    message,
) -> None:
    artifact, reference, graph = source_artifact
    artifact = deepcopy(artifact)
    mutate(artifact)
    with pytest.raises(VerificationError, match=message):
        verification.replay_verification_artifact(
            artifact,
            reference_path=reference,
            graph_path=graph,
            workload_name="identity",
            workload_parameters={},
            required_tolerance=verification.TolerancePolicy(
                atol=0.0,
                rtol=0.0,
            ),
            lifecycle=ir_lifecycle(IRKind.ATEN),
        )


def test_replay_rejects_subject_digest_mismatches(source_artifact) -> None:
    artifact, reference, graph = source_artifact
    changed = deepcopy(artifact)
    changed["subject"][0]["digest"]["sha256"] = "0" * 64
    with pytest.raises(VerificationError, match="reference SHA-256 mismatch"):
        verification.replay_verification_artifact(
            changed,
            reference_path=reference,
            graph_path=graph,
            workload_name="identity",
            workload_parameters={},
            required_tolerance=verification.TolerancePolicy(atol=0, rtol=0),
            lifecycle=ir_lifecycle(IRKind.ATEN),
        )
    changed = deepcopy(artifact)
    changed["subject"][1]["digest"]["sha256"] = "0" * 64
    with pytest.raises(VerificationError, match="graph SHA-256 mismatch"):
        verification.replay_verification_artifact(
            changed,
            reference_path=reference,
            graph_path=graph,
            workload_name="identity",
            workload_parameters={},
            required_tolerance=verification.TolerancePolicy(atol=0, rtol=0),
            lifecycle=ir_lifecycle(IRKind.ATEN),
        )


def test_execution_identity_and_operation_resolution(monkeypatch) -> None:
    assert verification._execution_identity("cpu") == {
        "device_type": "cpu",
        "backend": "cpu",
        "device": "cpu",
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(VerificationError, match="device is unavailable"):
        verification._execution_identity("cu" + "da:0")
