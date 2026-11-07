"""Observe one current SOLAR path, retaining stage failures without repairs."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import torch

from scripts.internal.solar.base01_observations import AccountingStatus
from scripts.internal.solar.base01_oracles import (
    f_expectations,
    resource_expectations,
)
from scripts.internal.solar.base01_records import (
    PROFILE,
    file_inventory,
    require_cpu_isolation,
    write_record,
)
from scripts.internal.solar.base01_samples import get_sample
from solar.artifacts import sha256_file
from solar.contracts import AnalysisRequest, ConversionRequest, SolarStage
from solar.errors import SolarError
from solar.ir.contracts import IRGraphArtifact, IRPath
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)
from solar.pipeline.stages import analyze_request_graph, workflow_reason_code
from solar.rocm.architecture import ArchitectureProfile

ERROR_MESSAGE_LIMIT = 4096


def input_identities(
    inputs: tuple[torch.Tensor, ...],
) -> list[dict[str, object]]:
    """Bind actual CPU trace inputs, including index and mixed-dtype bytes."""
    return [
        {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "device": str(value.device),
            "sha256": hashlib.sha256(
                value.detach().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
        for value in inputs
    ]


def observe(sample_id: str, ir_path: IRPath, output: Path) -> dict[str, object]:
    """Keep semantic verification mandatory; collect non-published CPU counts."""
    require_cpu_isolation()
    torch.set_num_threads(1)
    sample = get_sample(sample_id)
    source = Path(__file__).with_name("base01_samples.py")
    conversion = sample.request(ir_path, sha256_file(source))
    readiness = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=output / "conversion",
            require_verified_audit=False,
        )
    )
    record: dict[str, object] = {
        "sample": asdict(sample),
        "ir_path": ir_path.value,
        "policy": asdict(conversion.verification),
        "trace_seed": conversion.trace_seed,
        "trace_inputs": input_identities(sample.inputs(conversion.trace_seed)),
        "reference_sha256": conversion.reference_sha256,
        "readiness": readiness.to_dict(),
        "accounting_status": AccountingStatus.NOT_RUN,
        "publication_eligible": False,
        "real_mapper": "not_run",
        "expected_math": f_expectations().get(sample_id, {}),
        "expected_resource_work": resource_expectations().get(sample_id, {}),
        "accounting_request": {
            "precision": "fp32",
            "require_orojenesis": False,
            "require_verified_audit": False,
            "profile_sha256": sha256_file(PROFILE),
            "scope": "cpu_observation_only",
        },
    }
    if readiness.ready:
        record.update(observe_accounting(conversion, sample_id, output))
    record["artifacts"] = file_inventory(output)
    write_record(output / "observation.json", record)
    return record


def observe_accounting(
    conversion: ConversionRequest, sample_id: str, output: Path
) -> dict[str, object]:
    """Run the existing counter with no mapper or qualification promotion."""
    graph_name = (
        "aten_graph.yaml"
        if conversion.ir_path is IRPath.MAKE_FX_ATEN
        else "einsum_graph.yaml"
    )
    graph = IRGraphArtifact(
        path=output / "conversion" / graph_name,
        kind=conversion.ir_path.ir_kind,
    )
    request = AnalysisRequest(
        conversion=conversion,
        architecture=PROFILE,
        output_dir=output,
        precision="fp32",
        require_orojenesis=False,
        require_verified_audit=False,
    )
    try:
        result = analyze_request_graph(
            request,
            ArchitectureProfile.load(PROFILE),
            output,
            graph,
        )
    except SolarError as exc:
        return {
            "accounting_status": AccountingStatus.FAILED,
            "reason_code": workflow_reason_code(
                SolarStage.FORMAL_ANALYSIS, exc
            ),
            "message": str(exc)[:ERROR_MESSAGE_LIMIT],
        }
    expected = f_expectations().get(sample_id, {})
    totals = result["total"]
    comparison = {
        key: {
            "expected": value,
            "observed": totals.get(key),
            "matches": totals.get(key) == value,
        }
        for key, value in expected.items()
    }
    for resource, modes in resource_expectations().get(sample_id, {}).items():
        for mode, value in modes.items():
            observed = (
                totals.get("resource_work", {}).get(resource, {}).get(mode, 0)
            )
            comparison[f"resource_work.{resource}.{mode}"] = {
                "expected": value,
                "observed": observed,
                "matches": observed == value,
            }
    return {
        "accounting_status": AccountingStatus.COMPLETED,
        "total": totals,
        "analysis_metadata": result["metadata"],
        "math_comparison": comparison,
    }


def main() -> int:
    """Collect a single isolated slot; the parent retains abnormal exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--ir-path", required=True, choices=list(IRPath))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observe(args.sample, IRPath(args.ir_path), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
