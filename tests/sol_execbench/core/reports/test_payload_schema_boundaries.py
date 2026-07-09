from __future__ import annotations

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[4] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "sol_execbench"


RAW_PAYLOAD_INFRASTRUCTURE = {
    "sol_execbench.core.data.json_utils",
    "sol_execbench.core.data.solution_models",
}

RAW_PAYLOAD_ARTIFACT_BOUNDARIES = {
    "sol_execbench.core.evidence.baseline_export",
    "sol_execbench.core.reports.evaluation_stability.builder",
    "sol_execbench.core.bench.output_allocation",
    "sol_execbench.core.bench.profile_summary.artifacts",
    "sol_execbench.core.bench.static_kernel.artifacts",
    "sol_execbench.core.dataset.low_precision",
    "sol_execbench.core.dataset.migration.artifacts",
    "sol_execbench.core.scoring.amd_bound_graph.fx_helpers",
    "sol_execbench.core.scoring.amd_hardware_models",
    "sol_execbench.core.platform.arch_capabilities",
}

RAW_PAYLOAD_PARSER_BOUNDARIES = {
    "sol_execbench.core.dataset.paper_denominator.sources",
    "sol_execbench.core.scoring.amd_score.sidecar_parsing",
    "sol_execbench.core.scoring.amd_sol.v2_parsing",
    "sol_execbench.core.scoring.baseline_artifact",
    "sol_execbench.core.scoring.parsing_utils",
    "sol_execbench.core.scoring.solar_derivation.parse_root",
    "sol_execbench.core.scoring.solar_derivation.parse_utils",
}

RAW_PAYLOAD_ALLOWLIST = (
    RAW_PAYLOAD_INFRASTRUCTURE
    | RAW_PAYLOAD_ARTIFACT_BOUNDARIES
    | RAW_PAYLOAD_PARSER_BOUNDARIES
)

GET_CALL_BUDGET = {
    "sol_execbench.core.dataset.paper_denominator.stages": 45,
    "sol_execbench.core.reports.evaluation_stability.builder": 25,
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


def _get_call_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        )
        if count:
            counts[_module_name(path)] = count
    return counts


def test_raw_payload_shape_checks_stay_in_allowlisted_boundary_modules() -> None:
    modules_with_shape_checks = {
        _module_name(path)
        for path in PACKAGE_ROOT.rglob("*.py")
        if _has_raw_payload_shape_checks(path)
    }

    unexpected = sorted(modules_with_shape_checks - RAW_PAYLOAD_ALLOWLIST)
    stale_allowlist_entries = sorted(RAW_PAYLOAD_ALLOWLIST - modules_with_shape_checks)

    assert unexpected == []
    assert stale_allowlist_entries == []


def test_get_call_hotspots_do_not_regress_in_business_modules() -> None:
    modules_with_counts = _get_call_counts()
    over_budget = {
        module: count
        for module, count in modules_with_counts.items()
        if count >= 20 and count > GET_CALL_BUDGET.get(module, 19)
    }

    assert over_budget == {}
