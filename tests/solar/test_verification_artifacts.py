from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from solar.verification import einsum as verification
from solar.verification.einsum import VerificationError


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
        "schema_version": 3,
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
    tmp_path: Path, verification_files
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
        atol=0.0,
        rtol=0.0,
    )

    assert yaml.safe_load(output.read_text()) == artifact
    assert len(artifact["predicate"]["cases"]) == 9
    verification.replay_verification_artifact(
        artifact,
        reference_path=reference,
        graph_path=graph,
        workload_name="identity",
        workload_parameters={},
        atol=0.0,
        rtol=0.0,
    )


def test_callable_verification_writes_hash_bound_attestation(
    tmp_path: Path, verification_files
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
        atol=0.0,
        rtol=0.0,
    )

    assert artifact["subject"][0]["digest"]["sha256"] == "a" * 64
    assert len(artifact["predicate"]["results"]) == 9
    assert yaml.safe_load(output.read_text()) == artifact


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (
            verification.create_verification_artifact,
            {"seeds": (1, 1, 2)},
            "at least three seeds",
        ),
        (
            verification.create_verification_artifact,
            {"patterns": ("random", "zeros")},
            "boundary patterns",
        ),
        (
            verification.verify_callable_conversion,
            {"reference_sha256": "BAD"},
            "lowercase SHA-256",
        ),
        (
            verification.verify_callable_conversion,
            {"seeds": (1, 1, 2)},
            "at least three seeds",
        ),
        (
            verification.verify_callable_conversion,
            {"patterns": ("random", "zeros")},
            "boundary patterns",
        ),
    ],
)
def test_artifact_creation_rejects_weak_case_sets(
    tmp_path: Path, verification_files, function, kwargs, message
) -> None:
    reference, graph = verification_files
    common = {
        "graph_path": graph,
        "output_path": tmp_path / "out.yaml",
        "atol": 0.0,
        "rtol": 0.0,
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
    common = {
        "reference": lambda *values: values[0],
        "input_factory": lambda parameters, device: [torch.ones(2, 2), 4],
        "cases": cases,
        "atol": 0.0,
        "rtol": 0.0,
        "required_matched_ratio": 1.0,
        "max_error_cap": None,
        "allow_negative_inf": False,
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
    assert verification._pattern_inputs(source, "random") == source
    zeros = verification._pattern_inputs(source, "zeros")
    assert torch.equal(zeros[0], torch.zeros(3))
    boundary = verification._pattern_inputs(source, "boundary")
    assert boundary[0].tolist() == [-1.0, 0.0, 1.0]
    assert boundary[1].tolist() == [False, True, False]
    assert boundary[2].tolist() == [0, 0, 0]
    assert boundary[3] == "value"
    with pytest.raises(VerificationError, match="unknown verification input pattern"):
        verification._pattern_inputs(source, "missing")


def test_assert_close_reports_success_and_all_failure_modes() -> None:
    assert verification._assert_close(torch.ones(2), torch.ones(2), 0, 0) == {
        "max_abs_error": 0.0,
        "matched_ratio": 1.0,
    }
    assert (
        verification._assert_close(
            torch.tensor([-torch.inf]),
            torch.tensor([-torch.inf]),
            0,
            0,
            allow_negative_inf=True,
        )["matched_ratio"]
        == 1.0
    )
    assert (
        verification._assert_close((torch.ones(1),), (torch.ones(1),), 0, 0)[
            "max_abs_error"
        ]
        == 0.0
    )
    assert (
        verification._assert_close({"x": torch.ones(1)}, {"x": torch.ones(1)}, 0, 0)[
            "max_abs_error"
        ]
        == 0.0
    )
    assert verification._assert_close("x", "x", 0, 0)["max_abs_error"] == 0.0

    cases = [
        (torch.ones(2), torch.ones(3), "shape mismatch"),
        (torch.ones(2), torch.ones(2, dtype=torch.float16), "dtype mismatch"),
        (torch.tensor([1]), torch.tensor([2]), "integer/bool tensor values differ"),
        (torch.tensor([torch.nan]), torch.tensor([torch.nan]), "non-finite"),
        (torch.zeros(2), torch.ones(2), "all-zero output"),
        ((torch.ones(1),), (torch.ones(1), torch.ones(1)), "arity mismatch"),
        ({"x": 1}, {"y": 1}, "mapping keys differ"),
        ("x", "y", "non-tensor output mismatch"),
    ]
    for actual, expected, message in cases:
        with pytest.raises(VerificationError, match=message):
            verification._assert_close(actual, expected, 0, 0)
    with pytest.raises(VerificationError, match="numerical mismatch"):
        verification._assert_close(torch.ones(2), torch.full((2,), 2.0), 0, 0)
    with pytest.raises(VerificationError, match="exceeds cap"):
        verification._assert_close(
            torch.tensor([1.1]),
            torch.tensor([1.0]),
            1,
            0,
            max_error_cap=0.01,
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
            atol=0.0,
            rtol=0.0,
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
                parameters_sha256="0" * 64
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
                slice(None), value["predicate"]["cases"][:2]
            ),
            "lacks the required cases",
        ),
        (
            lambda value: value["predicate"]["execution"].update(device_type="tpu"),
            "no supported replay device",
        ),
        (
            lambda value: value["predicate"]["execution"].update(backend="tpu"),
            "no execution backend identity",
        ),
    ],
)
def test_replay_rejects_untrusted_artifact_mutations(
    source_artifact, mutate, message
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
            atol=0.0,
            rtol=0.0,
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
            atol=0,
            rtol=0,
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
            atol=0,
            rtol=0,
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
    assert verification._resolve_torch_operation("torch.add") is torch.add
    with pytest.raises(VerificationError, match="cannot resolve"):
        verification._resolve_torch_operation("missing_operation")


VALID_HANDLER = """
def create_add_subgraph(node_id, node_data):
    return {
        node_id: {
            'type': 'add',
            'is_real_einsum': False,
            'tensor_names': {'inputs': ['left', 'right'], 'outputs': ['output']},
            'tensor_shapes': {
                'inputs': node_data['input_shapes'],
                'outputs': node_data['output_shapes'],
            },
            'tensor_dtypes': {
                'inputs': node_data['input_dtypes'],
                'outputs': node_data['output_dtypes'],
            },
        }
    }
"""


def test_generated_handler_verification_accepts_exact_subgraph() -> None:
    result = verification.verify_generated_handler(
        "add",
        VALID_HANDLER,
        {
            "input_shapes": [[2], [2]],
            "output_shapes": [[2]],
            "input_dtypes": ["torch.float32", "torch.float32"],
            "output_dtypes": ["torch.float32"],
        },
    )
    assert result["status"] == "passed"
    assert len(result["cases"]) == 3


@pytest.mark.parametrize(
    ("source", "node_data", "message"),
    [
        ("x = 1", {"input_shapes": [[2]]}, "does not define"),
        (
            "def create_add_subgraph(node_id, node_data): return {}",
            {"input_shapes": [[2]]},
            "empty subgraph",
        ),
        (
            "def create_add_subgraph(node_id, node_data): return {'x': {}}",
            {"input_shapes": []},
            "requires input shapes",
        ),
    ],
)
def test_generated_handler_verification_rejects_invalid_contracts(
    source, node_data, message
) -> None:
    with pytest.raises(VerificationError, match=message):
        verification.verify_generated_handler("add", source, node_data)
