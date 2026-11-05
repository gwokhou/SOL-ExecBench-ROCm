"""Run the L3-b stage-1 shape-domain campaign for R06 and focused samples.

The campaign validates the frozen R06 shape domain over its four workloads
plus two synthesized in-domain points, probes the pre-fixed negative
vocabulary, certifies structural stability across three instances, and
records the source domain inside a real analysis via the existing
``analysis_metadata`` seam. Acceptance is pre-fixed in
docs/internal/solar-l3b-shape-domain-acceptance-2026-11-05.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from scripts.internal.solar.l3b_samples import (
    assumption_probe_domain,
    get_l3b_sample,
)
from sol_execbench.core.bench.eval_runtime import load_reference_function
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.solar_bridge.formal_device import (
    FORMAL_ARCHITECTURE,
    device_execution_policy,
)
from sol_execbench.core.solar_bridge.models import (
    formal_precision_for_definition,
)
from sol_execbench.core.solar_bridge.semantic_metadata import (
    performance_analysis_metadata,
)
from sol_execbench.core.solar_bridge.workload_context import (
    load_solar_workload_context,
    solar_conversion_request,
)
from sol_execbench.core.timestamps import utc_timestamp
from solar.api import AnalysisRequest, analyze
from solar.artifacts import sha256_file
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import IRPath
from solar.shapes import (
    Assumption,
    AssumptionOp,
    DimExpr,
    ShapeDomain,
    ShapeReasonCode,
    ShapeSymbol,
    TensorBinding,
    TensorRole,
    bind_tensors,
    certify_structure,
    structural_projection,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
R06_PROBLEM = (
    REPO_ROOT / "problems" / "AMD_AKA" / "torch2flydsl" / "moe_topk_softmax"
)
R06_WORKLOAD_AXES = {
    "aka-moe_topk_softmax-w0": {"E": 256, "K": 8, "T": 64},
    "aka-moe_topk_softmax-w1": {"E": 256, "K": 8, "T": 1024},
    "aka-moe_topk_softmax-w2": {"E": 128, "K": 4, "T": 256},
    "aka-moe_topk_softmax-w3": {"E": 64, "K": 2, "T": 64},
}
SYNTHESIZED = {
    "boundary-max": {"T": 1024, "E": 256, "K": 8},
    "interior-non-corpus": {"T": 128, "E": 192, "K": 6},
}
NEGATIVE_PROBES = (
    ("out-of-range", {"T": 1030, "E": 256, "K": 8}, "out_of_range"),
    ("divisibility", {"T": 100, "E": 256, "K": 8}, "divisibility_violated"),
    (
        "assumption",
        {"T": 1024, "E": 64, "K": 100},
        "assumption_violated",
    ),
)


def r06_shape_domain() -> ShapeDomain:
    """Return the frozen R06 domain declaration (spec section 1)."""
    return ShapeDomain(
        domain_id="r06-moe-topk-softmax-shape-v1",
        symbols={
            "T": ShapeSymbol("T", 64, 1024, 64, provenance="definition.axes:T"),
            "E": ShapeSymbol("E", 64, 256, 64, provenance="definition.axes:E"),
            "K": ShapeSymbol("K", 2, 256, 2, provenance="definition.axes:K"),
        },
        tensors={
            "gating": TensorBinding(
                "gating",
                TensorRole.INPUT,
                (DimExpr.symbol("T"), DimExpr.symbol("E")),
            ),
            "bias": TensorBinding(
                "bias", TensorRole.INPUT, (DimExpr.symbol("E"),)
            ),
            "topk": TensorBinding("topk", TensorRole.INPUT, ()),
            "route_scale": TensorBinding("route_scale", TensorRole.INPUT, ()),
            "weights": TensorBinding(
                "weights",
                TensorRole.OUTPUT,
                (DimExpr.symbol("T"), DimExpr.symbol("K")),
            ),
            "ids": TensorBinding(
                "ids",
                TensorRole.OUTPUT,
                (DimExpr.symbol("T"), DimExpr.symbol("K")),
            ),
        },
        assumptions=(
            Assumption(
                DimExpr.symbol("K"),
                AssumptionOp.LE,
                DimExpr.symbol("E"),
            ),
        ),
    )


def _verdict_dict(verdict: Any) -> dict[str, Any]:
    return {
        "status": verdict.status.value,
        "detail": verdict.detail,
        "assignment": verdict.assignment,
    }


def run_domain_verdicts(domain: ShapeDomain) -> dict[str, Any]:
    """Validate the workload denominator, synthesized points, and negatives."""
    verdicts: dict[str, Any] = {"denominator": {}, "negatives": {}}
    for workload, axes in sorted(R06_WORKLOAD_AXES.items()):
        verdicts["denominator"][workload] = _verdict_dict(
            domain.validate_assignment(axes)
        )
    for name, axes in sorted(SYNTHESIZED.items()):
        verdicts["denominator"][name] = _verdict_dict(
            domain.validate_assignment(axes)
        )
    for name, axes, expected in NEGATIVE_PROBES:
        verdict = domain.validate_assignment(axes)
        verdicts["negatives"][name] = {
            **_verdict_dict(verdict),
            "expected": expected,
            "matches": verdict.status.value == expected,
        }
    verdicts["passed"] = all(
        item["status"] == "validated"
        for item in verdicts["denominator"].values()
    ) and all(item["matches"] for item in verdicts["negatives"].values())
    return verdicts


def _r06_inputs(module: Any, axes: dict[str, int]) -> tuple[Any, ...]:
    """Mirror the workload input kinds: scalars stay Python scalars."""
    gating = module.gen_gating({"E": axes["E"], "T": axes["T"]}, "cpu")[
        "gating"
    ]
    bias = torch.linspace(-1.0, 1.0, axes["E"], dtype=torch.float32)
    return (gating, bias, axes["K"], 1.0)


def run_structure_certification(
    domain: ShapeDomain, output: Path
) -> dict[str, Any]:
    """Certify R06 structural stability across three in-domain instances."""
    definition = json.loads((R06_PROBLEM / "definition.json").read_text())
    module, reference = load_reference_function(definition["reference"])
    instances = {
        "w0": R06_WORKLOAD_AXES["aka-moe_topk_softmax-w0"],
        "w1": R06_WORKLOAD_AXES["aka-moe_topk_softmax-w1"],
        "interior-non-corpus": SYNTHESIZED["interior-non-corpus"],
    }
    projections: dict[str, list[dict[str, Any]]] = {}
    for key, axes in instances.items():
        inputs = _r06_inputs(module, axes)
        artifact = extract_operator_graph(
            reference,
            inputs,
            device="cpu",
            output_dir=output / key,
            name=key,
            extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
        )
        document = yaml.safe_load(artifact.path.read_text())
        projections[key] = structural_projection(document)
    r06_verdict = certify_structure(projections, instances)

    branch = get_l3b_sample("shape_branch")
    generator = torch.Generator(device="cpu").manual_seed(200)
    branch_projections = {}
    for key, size in (("small", 64), ("large", 128)):
        value = torch.randn((size,), generator=generator)
        artifact = extract_operator_graph(
            branch.reference,
            (value,),
            device="cpu",
            output_dir=output / f"branch-{key}",
            name=key,
            extraction_kind=IRPath.MAKE_FX_ATEN.extraction_kind,
        )
        document = yaml.safe_load(artifact.path.read_text())
        branch_projections[key] = structural_projection(document)
    branch_verdict = certify_structure(
        branch_projections, {"small": {"N": 64}, "large": {"N": 128}}
    )
    return {
        "r06": {
            "instances": instances,
            **_verdict_dict(r06_verdict),
        },
        "shape_branch_sample": _verdict_dict(branch_verdict),
        "passed": (
            r06_verdict.validated
            and branch_verdict.status is ShapeReasonCode.SHAPE_BRANCH_REJECTED
        ),
    }


def run_focused_vocabulary(output: Path) -> dict[str, Any]:
    """Evaluate the focused sample vocabulary at campaign level."""
    results: dict[str, Any] = {}

    shared = get_l3b_sample("shared_and_derived")
    inputs = shared.inputs(200)
    tensor_out = shared.reference(*inputs)
    results["shared_and_derived"] = _verdict_dict(
        bind_tensors(
            shared.domain,
            {
                "left": tuple(inputs[0].shape),
                "right": tuple(inputs[1].shape),
                "y": tuple(tensor_out.shape),
            },
        )
    )

    nonzero = get_l3b_sample("nonzero_interval")
    value = nonzero.inputs(200)[0]
    observed = nonzero.reference(value)
    nonzero_verdict = bind_tensors(
        nonzero.domain,
        {
            "value": tuple(value.shape),
            "y": tuple(observed.shape),
        },
    )
    upper = nonzero.domain.validate_assignment({"N": 64, "nnz": 65})
    results["nonzero_interval"] = {
        "observed": _verdict_dict(nonzero_verdict),
        "exceeded_probe": {
            **_verdict_dict(upper),
            "matches": upper.status is ShapeReasonCode.INTERVAL_EXCEEDED,
        },
    }

    probe = assumption_probe_domain()
    violated = probe.validate_assignment({"N": 4, "M": 3})
    results["assumption_probe"] = {
        **_verdict_dict(violated),
        "matches": violated.status is ShapeReasonCode.ASSUMPTION_VIOLATED,
    }

    mutation = get_l3b_sample("alias_mutation")
    mutation_inputs = mutation.inputs(200)
    mutation_out = mutation.reference(*mutation_inputs)
    results["alias_mutation_bind"] = _verdict_dict(
        bind_tensors(
            mutation.domain,
            {
                "value": tuple(mutation_inputs[0].shape),
                "y": tuple(mutation_out.shape),
            },
        )
    )
    results["passed"] = (
        results["shared_and_derived"]["status"] == "validated"
        and results["nonzero_interval"]["observed"]["status"] == "validated"
        and results["nonzero_interval"]["exceeded_probe"]["matches"]
        and results["assumption_probe"]["matches"]
        and results["alias_mutation_bind"]["status"] == "validated"
    )
    return results


def _shape_domain_metadata(
    domain: ShapeDomain, axes: dict[str, int]
) -> dict[str, Any]:
    shapes = {
        name: [
            dim if isinstance(dim, int) else dim.evaluate(axes)
            for dim in binding.dims
        ]
        for name, binding in domain.tensors.items()
    }
    return {
        "domain_id": domain.domain_id,
        "domain_sha256": domain.stable_identity(),
        "assignment": dict(sorted(axes.items())),
        "tensor_shapes": shapes,
    }


def _analysis_request(
    context: Any, output: Path, orojenesis_home: Path, metadata: dict[str, Any]
) -> AnalysisRequest:
    return AnalysisRequest(
        conversion=solar_conversion_request(
            context, "cpu", IRPath.MAKE_FX_ATEN
        ),
        architecture=FORMAL_ARCHITECTURE,
        output_dir=output,
        precision=formal_precision_for_definition(context.definition),
        require_orojenesis=True,
        require_verified_audit=True,
        orojenesis_home=orojenesis_home,
        analysis_metadata=metadata,
        execution_policy=device_execution_policy("cpu", None, 60.0),
    )


def run_instance_analysis(
    domain: ShapeDomain, args: argparse.Namespace, output: Path
) -> dict[str, Any]:
    """Analyze R06 w0 with and without the shape-domain metadata block."""
    import shutil

    shutil.rmtree(output, ignore_errors=True)
    context = load_solar_workload_context(
        R06_PROBLEM, "aka-moe_topk_softmax-w0", "cpu"
    )
    axes = R06_WORKLOAD_AXES["aka-moe_topk_softmax-w0"]
    metadata = {
        **performance_analysis_metadata(context),
        "shape_domain": _shape_domain_metadata(domain, axes),
    }
    runs = {}
    for name, analysis_metadata in (
        ("with-domain", metadata),
        ("control", dict(performance_analysis_metadata(context))),
    ):
        request = _analysis_request(
            context,
            output / name,
            args.orojenesis_home,
            analysis_metadata,
        )
        result = analyze(request)
        runs[name] = result
    with_domain, control = runs["with-domain"], runs["control"]
    domain_document = yaml.safe_load(
        (output / "with-domain" / "solar-analysis.yaml").read_text()
    )
    control_document = yaml.safe_load(
        (output / "control" / "solar-analysis.yaml").read_text()
    )
    recorded = domain_document.get("metadata", {}).get("shape_domain", {})
    with_domain_document = json.loads(json.dumps(domain_document))
    with_domain_document["metadata"].pop("shape_domain")
    invariance_diff = _first_difference(with_domain_document, control_document)
    manifest = yaml.safe_load(
        (output / "with-domain" / "manifest.yaml").read_text()
    )
    recorded_artifacts = {
        item["path"]: item["sha256"] for item in manifest["artifacts"]
    }
    analysis_sha = sha256_file(output / "with-domain" / "solar-analysis.yaml")
    verdict = {
        "with_domain_status": with_domain.status.value,
        "control_status": control.status.value,
        "metadata_present": bool(recorded),
        "domain_sha256_matches": (
            recorded.get("domain_sha256") == domain.stable_identity()
        ),
        "manifest_binds_analysis": (
            recorded_artifacts.get("solar-analysis.yaml") == analysis_sha
        ),
        "invariance_diff": invariance_diff,
        "passed": (
            with_domain.status.value == "analyzed"
            and control.status.value == "analyzed"
            and bool(recorded)
            and recorded.get("domain_sha256") == domain.stable_identity()
            and recorded_artifacts.get("solar-analysis.yaml") == analysis_sha
            and invariance_diff is None
        ),
    }
    atomic_write_json_value(output / "instance-analysis-verdict.json", verdict)
    return verdict


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys: dict[str, Any] = left
        right_keys: dict[str, Any] = right
        for key in sorted(set(left_keys) | set(right_keys)):
            if key not in left_keys or key not in right_keys:
                return f"{path}.{key}"
            deeper = _first_difference(left[key], right[key], f"{path}.{key}")
            if deeper is not None:
                return deeper
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}[len]"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            deeper = _first_difference(a, b, f"{path}[{index}]")
            if deeper is not None:
                return deeper
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "git_commit": revision,
        "r06_definition_sha256": sha256_file(R06_PROBLEM / "definition.json"),
        "mapper_sha256": sha256_file(
            args.orojenesis_home / "bin" / "timeloop-mapper"
        ),
        "torch_version": torch.__version__,
    }


def main() -> int:
    """Run the full L3-b campaign and write the pre-fixed gate summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--orojenesis-home",
        type=Path,
        default=Path(
            "/home/guohao/PyCharmMiscProject/CrossSOL-Eval/out/"
            "paper-a-local-20261005-v8/infrastructure-amd-v8/"
            "prerequisites/external/amd-orojenesis"
        ),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    domain = r06_shape_domain()
    round_trip = ShapeDomain.from_dict(domain.to_dict())
    verdicts = run_domain_verdicts(domain)
    atomic_write_json_value(
        args.output_root / "r06-domain-verdicts.json", verdicts
    )
    structure = run_structure_certification(
        domain, args.output_root / "structure"
    )
    atomic_write_json_value(args.output_root / "structure-cert.json", structure)
    focused = run_focused_vocabulary(args.output_root / "focused")
    atomic_write_json_value(
        args.output_root / "focused-vocabulary.json", focused
    )
    instance = run_instance_analysis(
        domain, args, args.output_root / "instance-analysis"
    )
    summary = {
        "schema_version": "sol_execbench.solar_l3b_summary.v1",
        "generated_at": utc_timestamp(),
        "identity": _identity(args),
        "domain_sha256": domain.stable_identity(),
        "round_trip_identity_stable": (
            round_trip.stable_identity() == domain.stable_identity()
        ),
        "gate": {
            "domain_verdicts": verdicts["passed"],
            "structure": structure["passed"],
            "focused": focused["passed"],
            "instance_analysis": instance["passed"],
        },
        "instance_verdict": instance,
    }
    summary["passed"] = (
        verdicts["passed"]
        and structure["passed"]
        and focused["passed"]
        and instance["passed"]
        and summary["round_trip_identity_stable"]
    )
    atomic_write_json_value(args.output_root / "l3b-summary.json", summary)
    print(json.dumps({"passed": summary["passed"], "gate": summary["gate"]}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
