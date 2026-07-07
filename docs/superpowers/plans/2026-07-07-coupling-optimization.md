# Coupling Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce avoidable internal import coupling while preserving existing SOL ExecBench ROCm behavior.

**Architecture:** Add static import-boundary tests, then make dependency direction explicit with minimal code changes. The SOLAR evidence model should no longer import coverage implementation code; coverage helpers should depend on model dataclasses instead.

**Tech Stack:** Python 3.12, pytest, `ast`, Ruff, existing `src/sol_execbench` package layout.

---

## File Structure

- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`
  - Add reusable AST import graph helpers and assertions for source-layer boundaries.
- Modify: `src/sol_execbench/core/scoring/solar_derivation_models.py`
  - Remove the lazy import of `solar_derivation_coverage` from `SolarDerivationEvidence.to_dict()`.
  - Add optional `coverage_summary` and `aggregate_status` fields to store precomputed derived metadata.
- Modify: `src/sol_execbench/core/scoring/solar_derivation_builders.py`
  - Populate `coverage_summary` and `aggregate_status` when constructing `SolarDerivationEvidence`.
- Modify only if tests prove safe: `src/sol_execbench/core/dataset/__init__.py`
  - Remove unused package-level re-exports that are not used by internal callers.
- Modify only if tests prove safe: `src/sol_execbench/core/scoring/__init__.py`
  - Remove unused package-level re-exports that are not used by internal callers.

---

### Task 1: Add Import Boundary Regression Tests

**Files:**
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Write the failing test**

Append these helpers and tests to `tests/sol_execbench/test_cli_module_boundaries.py`:

```python
import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "sol_execbench"


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


def _resolve_imported_modules(
    module: str,
    node: ast.Import | ast.ImportFrom,
    modules: dict[str, Path],
) -> set[str]:
    names: list[str] = []
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif node.level:
        package_parts = module.split(".")[:-1]
        if modules[module].name == "__init__.py":
            package_parts = module.split(".")
        base_parts = package_parts[: max(0, len(package_parts) - node.level + 1)]
        if node.module:
            base_parts.extend(node.module.split("."))
        base = ".".join(base_parts)
        names = [base]
        names.extend(
            f"{base}.{alias.name}" if base else alias.name
            for alias in node.names
            if alias.name != "*"
        )
    else:
        base = node.module or ""
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
    allowed = {
        (
            "sol_execbench.core.dataset.amd_score_reports",
            "sol_execbench.core.scoring.amd_hardware_models",
        ),
        (
            "sol_execbench.core.dataset.amd_score_reports",
            "sol_execbench.core.scoring.amd_score",
        ),
        (
            "sol_execbench.core.dataset.amd_score_reports",
            "sol_execbench.core.scoring.amd_sol",
        ),
        (
            "sol_execbench.core.dataset.amd_score_reports",
            "sol_execbench.core.scoring.amd_sol_v2",
        ),
        (
            "sol_execbench.core.dataset.amd_score_reports",
            "sol_execbench.core.scoring.baseline_artifact",
        ),
        (
            "sol_execbench.core.dataset.amd_score_reports",
            "sol_execbench.core.scoring.solar_derivation",
        ),
        (
            "sol_execbench.core.dataset.cli_execution",
            "sol_execbench.core.bench.io",
        ),
        (
            "sol_execbench.core.dataset.cli_execution",
            "sol_execbench.core.bench.stderr",
        ),
        (
            "sol_execbench.core.dataset.runner",
            "sol_execbench.core.bench.config",
        ),
        (
            "sol_execbench.core.dataset.runner",
            "sol_execbench.core.bench.rocm_profiler",
        ),
        (
            "sol_execbench.core.dataset.runner",
            "sol_execbench.core.scoring.amd_score",
        ),
        (
            "sol_execbench.core.dataset.runner",
            "sol_execbench.core.scoring.baseline_artifact",
        ),
        (
            "sol_execbench.core.bench.agent_feedback",
            "sol_execbench.core.dataset.checksums",
        ),
        (
            "sol_execbench.core.bench.profile_summary",
            "sol_execbench.core.dataset.checksums",
        ),
        (
            "sol_execbench.core.bench.static_kernel_artifacts",
            "sol_execbench.core.dataset.checksums",
        ),
        (
            "sol_execbench.core.scoring.amd_bound_sanity",
            "sol_execbench.core.dataset.manifest",
        ),
    }
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

    assert cross_domain_edges == sorted(allowed)


def test_no_internal_two_node_import_cycles() -> None:
    allowed = {
        ("sol_execbench.cli", "sol_execbench.cli.main"),
    }
    edges = _internal_import_edges()
    cycles = {
        tuple(sorted((source, target)))
        for source, target in edges
        if (target, source) in edges
    }

    assert cycles == allowed
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py::test_no_internal_two_node_import_cycles -q
```

Expected: `FAIL` showing the extra cycle between `solar_derivation_coverage` and `solar_derivation_models`.

- [ ] **Step 3: Do not change production code yet**

Leave the failing test in place and proceed to Task 2.

---

### Task 2: Break the SOLAR Model/Coverage Cycle

**Files:**
- Modify: `src/sol_execbench/core/scoring/solar_derivation_models.py`
- Modify: `src/sol_execbench/core/scoring/solar_derivation_builders.py`
- Test: `tests/sol_execbench/test_cli_module_boundaries.py`
- Test: `tests/sol_execbench/test_solar_derivation_contract.py`
- Test: `tests/sol_execbench/test_solar_derivation_evidence.py`

- [ ] **Step 1: Inspect builder construction sites**

Run:

```bash
rg -n "SolarDerivationEvidence\\(" src tests
```

Expected: construction sites include `solar_derivation_builders.py` and tests.

- [ ] **Step 2: Update `SolarDerivationEvidence` to store derived metadata**

In `src/sol_execbench/core/scoring/solar_derivation_models.py`, replace the `SolarDerivationEvidence` dataclass with this version:

```python
@dataclass(frozen=True)
class SolarDerivationEvidence:
    """Stable internal SOLAR derivation evidence sidecar."""

    definition: str
    workload_uuid: str
    groups: tuple[SolarSemanticGroupEvidence, ...]
    tensors: tuple[SolarTensorEvidence, ...]
    warnings: tuple[str, ...]
    source_boundary: dict[str, bool]
    coverage_summary: SolarCoverageSummary
    aggregate_status: SolarAggregateStatus
    schema_version: str = SOLAR_DERIVATION_SCHEMA_VERSION
    derived: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "derived": self.derived,
            "definition": self.definition,
            "workload_uuid": self.workload_uuid,
            "groups": [group.to_dict() for group in self.groups],
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "warnings": list(self.warnings),
            "source_boundary": dict(self.source_boundary),
            "coverage_summary": self.coverage_summary.to_dict(),
            "aggregate_status": self.aggregate_status.to_dict(),
        }
```

- [ ] **Step 3: Update the builder to compute the metadata**

In `src/sol_execbench/core/scoring/solar_derivation_builders.py`, ensure the module imports these helpers from `solar_derivation_coverage`:

```python
from sol_execbench.core.scoring.solar_derivation_coverage import (
    _aggregate_status_for_groups,
    _coverage_for_groups,
    _default_source_boundary,
    _derivation_warnings,
    _status_for_confidence,
)
```

Then update the `SolarDerivationEvidence(...)` construction to pass:

```python
coverage_summary=_coverage_for_groups(tuple(groups)),
aggregate_status=_aggregate_status_for_groups(tuple(groups), warnings),
```

Use local tuple variables if that keeps the construction DRY:

```python
group_evidence = tuple(groups)
tensor_evidence = tuple(tensors)
warnings = _derivation_warnings(graph, estimates)
return SolarDerivationEvidence(
    definition=definition.name,
    workload_uuid=workload.uuid,
    groups=group_evidence,
    tensors=tensor_evidence,
    warnings=warnings,
    source_boundary=_default_source_boundary(),
    coverage_summary=_coverage_for_groups(group_evidence),
    aggregate_status=_aggregate_status_for_groups(group_evidence, warnings),
)
```

- [ ] **Step 4: Update direct test constructors if needed**

If tests construct `SolarDerivationEvidence` directly, pass minimal empty metadata:

```python
coverage_summary=SolarCoverageSummary(
    family_counts={},
    status_counts={"scored": 0, "degraded": 0, "unscored": 0},
    families=(),
    missing_patterns=(),
    unsupported_patterns=(),
    degraded_node_ids=(),
    unsupported_node_ids=(),
    estimated_node_ids=(),
    provenance=(),
),
aggregate_status=SolarAggregateStatus(
    status="unscored",
    score_eligible=False,
    reason="no semantic groups were derived",
    group_ids=(),
    node_ids=(),
    warnings=(),
),
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py::test_no_internal_two_node_import_cycles -q
uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_solar_derivation_evidence.py -q
```

Expected: all selected tests pass.

---

### Task 3: Verify Cross-Domain Boundary Allowlists

**Files:**
- Modify if needed: `tests/sol_execbench/test_cli_module_boundaries.py`
- Modify if needed: source files that introduce unexpected cross-domain edges.

- [ ] **Step 1: Run the new boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py::test_core_data_does_not_depend_on_higher_layers tests/sol_execbench/test_cli_module_boundaries.py::test_core_does_not_depend_on_cli tests/sol_execbench/test_cli_module_boundaries.py::test_cross_domain_imports_stay_explicitly_allowlisted -q
```

Expected: pass after Task 2. If it fails, the failure output gives the exact `(source, target)` edge.

- [ ] **Step 2: Fix unexpected edges with the smallest change**

If a failure reports a new edge, choose one of these exact fixes:

```python
# Prefer direct low-level imports when a package __init__ import creates extra edges.
from sol_execbench.core.dataset.manifest import DatasetManifest
```

instead of:

```python
from sol_execbench.core.dataset import DatasetManifest
```

Only add an allowlist entry when the dependency is an intentional orchestration edge.

- [ ] **Step 3: Re-run the same boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py::test_core_data_does_not_depend_on_higher_layers tests/sol_execbench/test_cli_module_boundaries.py::test_core_does_not_depend_on_cli tests/sol_execbench/test_cli_module_boundaries.py::test_cross_domain_imports_stay_explicitly_allowlisted -q
```

Expected: pass.

---

### Task 4: Conservatively Trim Package Re-Export Coupling

**Files:**
- Modify if proven safe: `src/sol_execbench/core/dataset/__init__.py`
- Modify if proven safe: `src/sol_execbench/core/scoring/__init__.py`
- Modify if needed: internal imports that can use direct module paths.

- [ ] **Step 1: Identify internal package-level imports**

Run:

```bash
rg -n "from sol_execbench\\.core\\.(dataset|scoring) import|import sol_execbench\\.core\\.(dataset|scoring)" src tests scripts
```

Expected: list of call sites that rely on package re-exports.

- [ ] **Step 2: Replace internal broad imports with direct imports**

For each internal source file reported by Step 1, replace package-level imports with direct module imports. Example:

```python
from sol_execbench.core.scoring.amd_score import score_amd_native_workload
```

instead of:

```python
from sol_execbench.core.scoring import score_amd_native_workload
```

Do not change tests that intentionally verify public package imports.

- [ ] **Step 3: Remove only unused re-exports**

After Step 2, inspect `src/sol_execbench/core/dataset/__init__.py` and `src/sol_execbench/core/scoring/__init__.py`. Remove an import and its `__all__` entry only when:

```bash
rg -n "from sol_execbench\\.core\\.(dataset|scoring) import <Name>|sol_execbench\\.core\\.(dataset|scoring)\\.<Name>" src tests scripts docs
```

returns no usage for that package-level name.

- [ ] **Step 4: Run package import and boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py -q
uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_dataset_inventory_readiness.py -q
```

Expected: all selected tests pass.

---

### Task 5: Final Verification

**Files:**
- Verify: source and test files changed by Tasks 1-4.

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_native_score.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run Ruff**

Run:

```bash
uv run --with ruff ruff check tests/sol_execbench/test_cli_module_boundaries.py src/sol_execbench/core/scoring/solar_derivation_models.py src/sol_execbench/core/scoring/solar_derivation_builders.py
```

Expected: no lint violations.

- [ ] **Step 3: Review git diff**

Run:

```bash
git diff -- src/sol_execbench/core/scoring/solar_derivation_models.py src/sol_execbench/core/scoring/solar_derivation_builders.py src/sol_execbench/core/dataset/__init__.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_cli_module_boundaries.py
```

Expected: diff only contains boundary tests, SOLAR cycle removal, and conservative import cleanup.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/sol_execbench/test_cli_module_boundaries.py src/sol_execbench/core/scoring/solar_derivation_models.py src/sol_execbench/core/scoring/solar_derivation_builders.py src/sol_execbench/core/dataset/__init__.py src/sol_execbench/core/scoring/__init__.py
git commit -s -m "refactor: reduce internal coupling"
```

Expected: commit succeeds with DCO sign-off. If `dataset/__init__.py` or `scoring/__init__.py` were not changed, omit them from `git add`.
