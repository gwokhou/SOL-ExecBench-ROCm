from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli import evaluation
from sol_execbench.cli import environment
from sol_execbench.cli import main as cli_main
from sol_execbench.cli import reporting
from sol_execbench.cli import sidecars
from sol_execbench.core.scoring import amd_score_reports

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "sol_execbench"
COUPLING_CHECK_SCRIPT = SOURCE_ROOT.parents[0] / "scripts" / "check_coupling.py"


def _is_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _internal_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module = ".".join(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        modules[module] = path
    return modules


def _resolve_import_from_base(
    module: str,
    node: ast.ImportFrom,
    modules: dict[str, Path],
) -> str:
    if node.level:
        package_parts = module.split(".")[:-1]
        if modules[module].name == "__init__.py":
            package_parts = module.split(".")
        base_parts = package_parts[: max(0, len(package_parts) - node.level + 1)]
        if node.module:
            base_parts.extend(node.module.split("."))
        return ".".join(base_parts)

    return node.module or ""


def _resolve_imported_modules(
    module: str,
    node: ast.Import | ast.ImportFrom,
    modules: dict[str, Path],
) -> set[str]:
    names: list[str] = []
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    else:
        base = _resolve_import_from_base(module, node, modules)
        names = [base]
        names.extend(
            f"{base}.{alias.name}" if base else alias.name
            for alias in node.names
            if alias.name != "*"
        )

    resolved: set[str] = set()
    for name in names:
        if not name.startswith("sol_execbench"):
            continue
        parts = name.split(".")
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if candidate in modules and candidate != module:
                resolved.add(candidate)
                break
    return resolved


def _internal_import_edges() -> set[tuple[str, str]]:
    modules = _internal_modules()
    edges: set[tuple[str, str]] = set()
    for module, path in modules.items():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                for imported in _resolve_imported_modules(module, node, modules):
                    edges.add((module, imported))
    return edges


def _strongly_connected_components(
    edges: set[tuple[str, str]],
) -> list[tuple[str, ...]]:
    modules = set(_internal_modules())
    graph = {module: set[str]() for module in modules}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set())

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def visit(module: str) -> None:
        nonlocal index
        indices[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)

        for target in graph[module]:
            if target not in indices:
                visit(target)
                lowlinks[module] = min(lowlinks[module], lowlinks[target])
            elif target in on_stack:
                lowlinks[module] = min(lowlinks[module], indices[target])

        if lowlinks[module] != indices[module]:
            return

        component: list[str] = []
        while True:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == module:
                break
        if len(component) > 1:
            components.append(tuple(sorted(component)))

    for module in sorted(graph):
        if module not in indices:
            visit(module)

    return sorted(components)


def _module_line_count(module: str) -> int:
    return len(_internal_modules()[module].read_text().splitlines())


def test_high_inbound_model_modules_keep_exact_internal_imports() -> None:
    expected_targets = {
        "sol_execbench.core.data.base_model": [],
        "sol_execbench.core.scoring.amd_bound_graph_models": [
            "sol_execbench.core.scoring.amd_bound_graph_enums",
            "sol_execbench.core.scoring.confidence",
        ],
        "sol_execbench.core.scoring.amd_hardware_models": [
            "sol_execbench.core.scoring.confidence",
        ],
        "sol_execbench.core.scoring.solar_derivation_models": [
            "sol_execbench.core.scoring.confidence",
            "sol_execbench.core.scoring.solar_derivation_coverage_models",
            "sol_execbench.core.scoring.solar_derivation_evidence_models",
        ],
    }
    edges = _internal_import_edges()

    observed = {
        module: sorted(target for source, target in edges if source == module)
        for module in expected_targets
    }

    assert observed == expected_targets


def test_core_data_does_not_depend_on_higher_layers() -> None:
    violations = sorted(
        (source, target)
        for source, target in _internal_import_edges()
        if _is_under(source, "sol_execbench.core.data")
        and not _is_under(target, "sol_execbench.core.data")
    )

    assert violations == []


def test_core_does_not_depend_on_cli() -> None:
    violations = sorted(
        (source, target)
        for source, target in _internal_import_edges()
        if _is_under(source, "sol_execbench.core")
        and _is_under(target, "sol_execbench.cli")
    )

    assert violations == []


def test_cross_domain_imports_stay_explicitly_allowlisted() -> None:
    allowed_with_rationale = {
        (
            "sol_execbench.core.dataset.amd_score_reports",
            "sol_execbench.core.scoring.amd_score_reports",
        ): "dataset compatibility module re-exports scoring report helpers",
        (
            "sol_execbench.core.dataset.cli_execution",
            "sol_execbench.core.bench.io",
        ): "dataset CLI execution loads benchmark result traces",
        (
            "sol_execbench.core.dataset.cli_execution",
            "sol_execbench.core.bench.stderr",
        ): "dataset CLI execution classifies benchmark stderr failures",
        (
            "sol_execbench.core.dataset.runner",
            "sol_execbench.core.bench.config",
        ): "dataset runner constructs benchmark configuration for runs",
        (
            "sol_execbench.core.dataset.runner",
            "sol_execbench.core.bench.rocm_profiler",
        ): "dataset runner wires optional ROCm profiler artifacts",
        (
            "sol_execbench.core.dataset.runner_scoring",
            "sol_execbench.core.scoring.amd_score",
        ): "dataset scoring bridge exposes AMD score types for run reporting",
        (
            "sol_execbench.core.dataset.runner_scoring",
            "sol_execbench.core.scoring.amd_score_reports",
        ): "dataset scoring bridge delegates AMD score report construction",
        (
            "sol_execbench.core.dataset.runner_scoring",
            "sol_execbench.core.scoring.baseline_artifact",
        ): "dataset scoring bridge accepts scoring baseline artifacts",
        (
            "sol_execbench.core.scoring.amd_bound_sanity_models",
            "sol_execbench.core.dataset.manifest",
        ): "AMD bound sanity report embeds dataset manifest checksums",
    }
    assert all(reason for reason in allowed_with_rationale.values())
    domains = (
        "sol_execbench.core.bench",
        "sol_execbench.core.dataset",
        "sol_execbench.core.scoring",
    )
    cross_domain_edges = sorted(
        (source, target)
        for source, target in _internal_import_edges()
        if any(_is_under(source, domain) for domain in domains)
        and any(_is_under(target, domain) for domain in domains)
        and next(domain for domain in domains if _is_under(source, domain))
        != next(domain for domain in domains if _is_under(target, domain))
    )

    assert cross_domain_edges == sorted(allowed_with_rationale)


def test_no_internal_import_cycles() -> None:
    edges = _internal_import_edges()

    assert _strongly_connected_components(edges) == []


def test_no_internal_two_node_import_cycles() -> None:
    edges = _internal_import_edges()
    cycles = {
        tuple(sorted((source, target)))
        for source, target in edges
        if (target, source) in edges
    }

    assert cycles == set()


def test_coupling_guardrail_script_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(COUPLING_CHECK_SCRIPT), "--json"],
        check=True,
        capture_output=True,
        text=True,
        cwd=SOURCE_ROOT.parents[0],
    )
    payload = json.loads(result.stdout)

    assert payload["cycles"] == []
    assert payload["facade_import_violations"] == []
    assert payload["limit_failures"] == []


def test_cli_main_import_fanout_stays_bounded() -> None:
    edges = _internal_import_edges()
    main_imports = sorted(
        target for source, target in edges if source == "sol_execbench.cli.main"
    )

    assert len(main_imports) <= 15


def test_p0_orchestrators_stay_bounded_after_refactor() -> None:
    edges = _internal_import_edges()
    imports_by_source = {
        source: sorted(target for edge_source, target in edges if edge_source == source)
        for source in (
            "sol_execbench.cli.evaluation.evaluator",
            "sol_execbench.driver.templates.eval_driver",
            "sol_execbench.core.bench.eval_workload_runner",
        )
    }

    assert len(imports_by_source["sol_execbench.cli.evaluation.evaluator"]) <= 8
    assert _module_line_count("sol_execbench.cli.evaluation.evaluator") <= 240
    assert imports_by_source["sol_execbench.driver.templates.eval_driver"] == [
        "sol_execbench.driver.eval_runtime_api"
    ]
    assert len(imports_by_source["sol_execbench.core.bench.eval_workload_runner"]) <= 8
    assert _module_line_count("sol_execbench.core.bench.eval_workload_runner") <= 320


def test_p1_orchestrators_stay_bounded_after_refactor() -> None:
    edges = _internal_import_edges()
    limits = {
        "sol_execbench.core.scoring.amd_score_reports": (160, 7),
        "sol_execbench.driver.problem_packager": (230, 7),
        "sol_execbench.core.bench.rocm_profiler": (180, 5),
    }
    observed = {
        module: (
            _module_line_count(module),
            len({target for source, target in edges if source == module}),
        )
        for module in limits
    }

    assert observed == {
        module: (line_count, fanout)
        for module, (line_limit, fanout_limit) in limits.items()
        for line_count, fanout in [observed[module]]
        if line_count <= line_limit and fanout <= fanout_limit
    }


def test_dataset_runner_no_longer_imports_scoring_bridge_directly() -> None:
    runner_imports = sorted(
        target
        for source, target in _internal_import_edges()
        if source == "sol_execbench.core.dataset.runner"
        and _is_under(target, "sol_execbench.core.scoring")
    )

    assert runner_imports == []


def test_p2_hotspots_stay_bounded_after_refactor() -> None:
    edges = _internal_import_edges()
    limits = {
        "sol_execbench.core.bench.agent_feedback": (180, 5),
        "sol_execbench.core.bench.profile_summary": (200, 6),
        "sol_execbench.core.bench.input_generation": (260, 5),
        "sol_execbench.core.scoring.amd_bound_graph_fx": (260, 6),
        "sol_execbench.core.bench.rocm_profiler": (180, 5),
        "sol_execbench.core.scoring.amd_sol_graph": (220, 6),
    }

    observed = {
        module: (_module_line_count(module), len({target for source, target in edges if source == module}))
        for module in limits
    }

    assert observed == {
        module: (line_count, fanout)
        for module, (line_limit, fanout_limit) in limits.items()
        for line_count, fanout in [observed[module]]
        if line_count <= line_limit and fanout <= fanout_limit
    }


def test_source_modules_do_not_import_core_or_data_facades() -> None:
    modules = _internal_modules()
    facade_roots = {"sol_execbench.core", "sol_execbench.core.data"}
    violations: list[tuple[str, str]] = []

    for module, path in modules.items():
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    (module, alias.name)
                    for alias in node.names
                    if alias.name in facade_roots
                )
                continue

            if not isinstance(node, ast.ImportFrom):
                continue

            base = _resolve_import_from_base(module, node, modules)
            if base in facade_roots:
                violations.extend((module, base) for _alias in node.names)

    assert sorted(set(violations)) == []


def test_source_modules_do_not_import_dataset_package_reexports() -> None:
    modules = _internal_modules()
    core_root = "sol_execbench.core"
    dataset_root = "sol_execbench.core.dataset"
    violations: list[tuple[str, str]] = []

    for module, path in modules.items():
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    (module, alias.name)
                    for alias in node.names
                    if alias.name == dataset_root
                )
                continue

            if not isinstance(node, ast.ImportFrom):
                continue

            base = _resolve_import_from_base(module, node, modules)
            if base == core_root:
                violations.extend(
                    (module, alias.name)
                    for alias in node.names
                    if alias.name == "dataset"
                )
                continue

            if base != dataset_root:
                continue

            for alias in node.names:
                imported = f"{dataset_root}.{alias.name}"
                if imported not in modules:
                    violations.append((module, alias.name))

    assert sorted(violations) == []


def test_dataset_package_facade_is_lazy() -> None:
    script = """
import sys

import sol_execbench.core.dataset as dataset

eager_modules = [
    "sol_execbench.core.dataset.inventory",
    "sol_execbench.core.dataset.low_precision",
    "sol_execbench.core.dataset.migration",
    "sol_execbench.core.dataset.paper_denominator",
    "sol_execbench.core.dataset.readiness",
]
loaded = [module for module in eager_modules if module in sys.modules]
if loaded:
    raise SystemExit(f"eagerly loaded dataset modules: {loaded}")

_ = dataset.DatasetShardPlan
if "sol_execbench.core.dataset.sharding" not in sys.modules:
    raise SystemExit("lazy access did not load sharding module")
late_loaded = [module for module in eager_modules if module in sys.modules]
if late_loaded:
    raise SystemExit(f"unrelated dataset modules loaded after sharding: {late_loaded}")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=SOURCE_ROOT,
    )


def test_source_modules_do_not_import_scoring_package_reexports() -> None:
    modules = _internal_modules()
    core_root = "sol_execbench.core"
    scoring_root = "sol_execbench.core.scoring"
    violations: list[tuple[str, str]] = []

    for module, path in modules.items():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    (module, alias.name)
                    for alias in node.names
                    if alias.name == scoring_root
                )
                continue

            if not isinstance(node, ast.ImportFrom):
                continue

            base = _resolve_import_from_base(module, node, modules)
            if base == core_root:
                violations.extend(
                    (module, alias.name)
                    for alias in node.names
                    if alias.name == "scoring"
                )
                continue

            if base != scoring_root:
                continue

            for alias in node.names:
                imported = f"{scoring_root}.{alias.name}"
                if imported not in modules:
                    violations.append((module, alias.name))

    assert sorted(violations) == []


def test_cli_main_import_does_not_eagerly_load_subcommand_modules() -> None:
    script = """
import sys

import sol_execbench.cli.main

eager_modules = [
    "sol_execbench.cli.baseline",
    "sol_execbench.cli.dataset",
    "sol_execbench.cli.metadata",
]
loaded = [module for module in eager_modules if module in sys.modules]
if loaded:
    raise SystemExit(f"eagerly loaded subcommand modules: {loaded}")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=SOURCE_ROOT,
    )


def test_cli_subcommand_completion_dispatches_to_subcommand_options() -> None:
    result = CliRunner().invoke(
        cli_main.cli,
        [],
        env={
            "_SOL_EXECBENCH_COMPLETE": "bash_complete",
            "COMP_WORDS": "sol-execbench contract --",
            "COMP_CWORD": "2",
        },
    )

    assert result.exit_code == 0, result.output
    assert "--json" in result.output
    assert "--definition" not in result.output


def test_cli_environment_helpers_live_outside_main_and_sidecars() -> None:
    assert environment._write_environment_snapshot_sidecar is not None
    assert environment._environment_snapshot_sidecar_path is not None
    assert environment.ENV_SNAPSHOT_ENABLE_ENV == "SOLEXECBENCH_ENV_SNAPSHOT"
    assert environment.ENV_SNAPSHOT_PATH_ENV == "SOLEXECBENCH_ENV_SNAPSHOT_PATH"

    for module in (cli_main, sidecars):
        for name in (
            "ENV_SNAPSHOT_ENABLE_ENV",
            "ENV_SNAPSHOT_PATH_ENV",
            "_environment_snapshot_sidecar_path",
            "_write_environment_snapshot_sidecar",
        ):
            assert not hasattr(module, name)


def test_cli_reporting_helpers_live_outside_main() -> None:
    assert reporting.print_traces_table is not None
    assert not hasattr(cli_main, "_print_traces_table")


def test_cli_sidecar_helpers_live_outside_main() -> None:
    assert sidecars._write_profile_sidecar is not None
    assert sidecars._write_profile_summary_sidecar is not None
    assert sidecars._write_static_evidence_sidecar is not None
    assert sidecars._write_agent_feedback_sidecar is not None
    assert sidecars._collect_static_evidence_for_cli is not None

    for name in (
        "_write_profile_sidecar",
        "_write_profile_summary_sidecar",
        "_write_static_evidence_sidecar",
        "_write_agent_feedback_sidecar",
        "_collect_static_evidence_for_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_evaluation_helpers_live_outside_main() -> None:
    assert evaluation._write_no_trace_diagnostics_sidecar is not None
    assert evaluation._timeout_output_text is not None
    assert evaluation._run_evaluation_command is not None
    assert evaluation._run_profiled_evaluation is not None

    for name in (
        "_write_no_trace_diagnostics_sidecar",
        "_timeout_output_text",
        "_run_evaluation_command",
        "_run_profiled_evaluation",
    ):
        assert not hasattr(cli_main, name)


def test_cli_metadata_commands_live_outside_main() -> None:
    from sol_execbench.cli import metadata

    assert metadata._contract_cli is not None
    assert metadata._doctor_cli is not None
    assert metadata._toolchain_cli is not None

    for name in (
        "_contract_cli",
        "_doctor_cli",
        "_toolchain_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_baseline_commands_live_outside_main() -> None:
    from sol_execbench.cli import baseline

    assert baseline._baseline_cli is not None
    assert baseline._baseline_export_cli is not None

    for name in (
        "_baseline_cli",
        "_baseline_export_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_dataset_commands_live_outside_main() -> None:
    from sol_execbench.cli import dataset

    assert dataset._dataset_cli is not None
    assert dataset._dataset_migrate_sol_cli is not None
    assert dataset._dataset_migrate_flashinfer_cli is not None

    for name in (
        "_dataset_cli",
        "_dataset_migrate_sol_cli",
        "_dataset_migrate_flashinfer_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_profile_sidecar_helpers_live_outside_main() -> None:
    from sol_execbench.cli import profile_sidecars

    profile_sidecar_names = (
        "_profile_output_directory",
        "_profile_sidecar_path",
        "_write_profile_sidecar",
        "_profile_summary_sidecar_path",
        "_write_profile_summary_sidecar",
        "_profile_summary_artifact_citations",
    )

    for name in profile_sidecar_names:
        assert getattr(profile_sidecars, name) is not None
        assert getattr(sidecars, name) is not None
        assert not hasattr(cli_main, name)


def test_cli_static_evidence_helpers_live_outside_main() -> None:
    from sol_execbench.cli import static_evidence

    assert static_evidence.STATIC_EVIDENCE_NONE == "none"
    assert static_evidence.STATIC_EVIDENCE_AUTO == "auto"
    assert static_evidence._static_evidence_directory is not None
    assert static_evidence._static_evidence_sidecar_path is not None
    assert static_evidence._static_evidence_summary is not None
    assert static_evidence._static_evidence_payload is not None
    assert static_evidence._write_static_evidence_sidecar is not None
    assert static_evidence._collect_static_evidence_for_cli is not None

    for name in (
        "STATIC_EVIDENCE_NONE",
        "STATIC_EVIDENCE_AUTO",
        "_static_evidence_directory",
        "_static_evidence_sidecar_path",
        "_static_evidence_summary",
        "_static_evidence_payload",
        "_write_static_evidence_sidecar",
        "_collect_static_evidence_for_cli",
    ):
        assert not hasattr(cli_main, name)


def test_cli_agent_feedback_sidecar_helpers_live_outside_main() -> None:
    from sol_execbench.cli import agent_feedback_sidecar

    assert agent_feedback_sidecar._agent_feedback_sidecar_path is not None
    assert agent_feedback_sidecar._write_agent_feedback_sidecar is not None
    assert agent_feedback_sidecar._agent_feedback_identity_fields is not None
    assert agent_feedback_sidecar._agent_feedback_run_id is not None
    assert agent_feedback_sidecar._agent_feedback_artifact_citations is not None

    for name in (
        "_agent_feedback_sidecar_path",
        "_write_agent_feedback_sidecar",
        "_agent_feedback_identity_fields",
        "_agent_feedback_run_id",
        "_agent_feedback_artifact_citations",
    ):
        assert not hasattr(cli_main, name)


def test_cli_problem_io_helpers_live_outside_main() -> None:
    from sol_execbench.cli import problem_io

    assert problem_io.ResolvedProblemInputs is not None
    assert problem_io._load_definition is not None
    assert problem_io._load_workloads is not None
    assert problem_io._load_solution is not None
    assert problem_io._load_config is not None
    assert problem_io._resolve_problem_dir is not None
    assert problem_io.resolve_problem_inputs is not None

    for name in (
        "ResolvedProblemInputs",
        "_load_definition",
        "_load_workloads",
        "_load_solution",
        "_load_config",
        "_resolve_problem_dir",
        "resolve_problem_inputs",
    ):
        assert not hasattr(cli_main, name)


def test_cli_compilation_helpers_live_outside_main() -> None:
    from sol_execbench.cli import compilation

    assert compilation.CompilePhaseResult is not None
    assert compilation.run_compile_phase is not None

    for name in (
        "CompilePhaseResult",
        "run_compile_phase",
    ):
        assert not hasattr(cli_main, name)


def test_cli_evaluation_runtime_helpers_live_outside_main() -> None:
    from sol_execbench.cli import evaluation_runtime

    assert evaluation_runtime.EvaluationRuntimeSuccess is not None
    assert evaluation_runtime.EvaluationRuntimeNoTraceFailure is not None
    assert evaluation_runtime.run_evaluation_runtime is not None

    for name in (
        "EvaluationRuntimeSuccess",
        "EvaluationRuntimeNoTraceFailure",
        "run_evaluation_runtime",
    ):
        assert not hasattr(cli_main, name)


def test_amd_score_report_sidecar_parsers_live_outside_orchestrator() -> None:
    from sol_execbench.core.scoring import amd_score_sidecar_parsing

    assert amd_score_sidecar_parsing.read_json_object is not None
    assert amd_score_sidecar_parsing.minimal_amd_sol_bound_v2_from_payload is not None
    assert amd_score_sidecar_parsing.minimal_solar_aggregate_from_payload is not None

    for name in (
        "_read_json_object",
        "_minimal_amd_sol_bound_v2_from_payload",
        "_minimal_solar_aggregate_from_payload",
    ):
        assert not hasattr(amd_score_reports, name)
