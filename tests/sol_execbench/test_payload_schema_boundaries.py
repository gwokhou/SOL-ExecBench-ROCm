from __future__ import annotations

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "sol_execbench"


RAW_PAYLOAD_ALLOWLIST = {
    # External trace and report readers.
    "sol_execbench.core.baseline_export",
    "sol_execbench.core.claim_upgrade",
    "sol_execbench.core.consistency",
    "sol_execbench.core.evaluation_stability",
    "sol_execbench.core.matrix_diff",
    "sol_execbench.core.trust_summary",
    # Core data compatibility parsers.
    "sol_execbench.core.data.json_utils",
    "sol_execbench.core.data.solution",
    # Bench artifact boundary readers.
    "sol_execbench.core.bench.output_allocation",
    "sol_execbench.core.bench.profile_summary_artifacts",
    "sol_execbench.core.bench.static_kernel_artifacts",
    # Dataset artifact and migration readers.
    "sol_execbench.core.dataset.low_precision",
    "sol_execbench.core.dataset.migration",
    "sol_execbench.core.dataset.paper_denominator_sources",
    "sol_execbench.core.dataset.parity_gap",
    "sol_execbench.core.dataset.profiler_timing_coverage",
    "sol_execbench.core.dataset.run_closure",
    # Scoring artifact and parser boundaries.
    "sol_execbench.core.scoring.amd_bound_graph_fx",
    "sol_execbench.core.scoring.amd_bound_sanity_builder",
    "sol_execbench.core.scoring.amd_bound_sanity_helpers",
    "sol_execbench.core.scoring.amd_score_sidecar_parsing",
    "sol_execbench.core.scoring.amd_hardware_models",
    "sol_execbench.core.scoring.amd_sol_v2_parsing",
    "sol_execbench.core.scoring.baseline_artifact",
    "sol_execbench.core.scoring.parsing_utils",
    "sol_execbench.core.scoring.solar_derivation_parse_root",
    "sol_execbench.core.scoring.solar_derivation_parse_utils",
}


def _module_name(path: Path) -> str:
    module = ".".join(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    return module.removesuffix(".__init__")


def _has_raw_payload_shape_checks(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            if len(node.args) == 2 and isinstance(node.args[1], ast.Name):
                if node.args[1].id == "dict":
                    return True
    return False


def test_raw_payload_shape_checks_stay_in_allowlisted_boundary_modules() -> None:
    modules_with_shape_checks = {
        _module_name(path)
        for path in PACKAGE_ROOT.rglob("*.py")
        if _has_raw_payload_shape_checks(path)
    }

    unexpected = sorted(modules_with_shape_checks - RAW_PAYLOAD_ALLOWLIST)

    assert unexpected == []
