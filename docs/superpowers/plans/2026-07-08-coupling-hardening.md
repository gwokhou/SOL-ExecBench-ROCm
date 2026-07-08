# Coupling Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce avoidable internal import coupling in `src/sol_execbench` while preserving current ROCm benchmark behavior and public schemas.

**Architecture:** Treat `core.data` and stable model modules as foundational leaf layers, keep CLI code as top-level orchestration, and make every intentional `bench` / `dataset` / `scoring` bridge explicit in boundary tests. Execute in small, test-first slices so high-inbound model files are protected before any cleanup touches package exports or CLI command wiring.

**Tech Stack:** Python 3.12, pytest, AST-based static import analysis, existing `src/sol_execbench` package layout, Ruff formatting.

---

## File Structure

- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`
  - Owns static import graph helpers and regression tests for allowed dependency direction.
- Modify: `src/sol_execbench/cli/main.py`
  - Keep Click root command registration only; move remaining command-specific helper imports out when tests identify a safe split.
- Modify only if tests prove compatibility: `src/sol_execbench/core/dataset/__init__.py`
  - Reduce internal reliance on package-level re-exports; preserve public names that tests or docs import.
- Modify only if tests prove compatibility: `src/sol_execbench/core/scoring/__init__.py`
  - Reduce internal reliance on package-level re-exports; preserve public names that tests or docs import.
- Create only if needed: `src/sol_execbench/cli/commands.py`
  - Optional small registration module if `_evaluate_cli` and command attachment need to be separated from `main.py`.
- Documentation check only: `docs/superpowers/specs/2026-07-07-coupling-optimization-design.md`
  - Update only if the implementation intentionally changes the documented allowed boundaries.

---

### Task 1: Solidify Import Boundary Regression Tests

**Files:**
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Add boundary tests for foundational layers**

Append these tests after `_internal_import_edges()` in `tests/sol_execbench/test_cli_module_boundaries.py`:

```python
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
```

- [ ] **Step 2: Add explicit cross-domain allowlist**

Append this test in the same file:

```python
def test_cross_domain_imports_stay_explicitly_allowlisted() -> None:
    allowed = {
        (
            "sol_execbench.core.dataset.amd_score_reports",
            "sol_execbench.core.scoring.amd_score_reports",
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
            "sol_execbench.core.scoring.amd_score_reports",
        ),
        (
            "sol_execbench.core.dataset.runner",
            "sol_execbench.core.scoring.baseline_artifact",
        ),
        (
            "sol_execbench.core.scoring.amd_bound_sanity_models",
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
```

- [ ] **Step 3: Add cycle guard**

Append this test in the same file:

```python
def test_no_internal_two_node_import_cycles_except_cli_entrypoint() -> None:
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

- [ ] **Step 4: Run boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS. If the cross-domain allowlist fails, inspect the reported edge and either remove the new dependency or add it only with a one-line comment explaining why it is a legitimate orchestration boundary.

- [ ] **Step 5: Commit**

```bash
git add tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Guard internal import boundaries"
```

---

### Task 2: Reduce CLI Main Import Fan-Out

**Files:**
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`
- Modify: `src/sol_execbench/cli/main.py`
- Create only if needed: `src/sol_execbench/cli/commands.py`

- [ ] **Step 1: Add a fan-out budget test for `cli.main`**

Append this test to `tests/sol_execbench/test_cli_module_boundaries.py`:

```python
def test_cli_main_import_fanout_stays_bounded() -> None:
    edges = _internal_import_edges()
    main_imports = sorted(
        target for source, target in edges if source == "sol_execbench.cli.main"
    )

    assert len(main_imports) <= 17
```

- [ ] **Step 2: Run the test and confirm the current baseline**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py::test_cli_main_import_fanout_stays_bounded -q
```

Expected: PASS with the current baseline. This test prevents `cli/main.py` from growing while later tasks lower the budget.

- [ ] **Step 3: Identify one safe CLI import to move**

Inspect `src/sol_execbench/cli/main.py` for imports that are only used to register subcommands or constants. Prefer moving command registration imports to `src/sol_execbench/cli/commands.py` only if the resulting code keeps `sol_execbench.cli:cli` behavior unchanged.

Use this initial shape if a new module is justified:

```python
# src/sol_execbench/cli/commands.py
from __future__ import annotations

import click

from . import baseline as cli_baseline
from . import dataset as cli_dataset
from . import evaluation_runtime as cli_evaluation_runtime
from . import metadata as cli_metadata


def register_subcommands(root: click.Group) -> None:
    root.add_command(cli_metadata._contract_cli)
    root.add_command(cli_metadata._doctor_cli)
    root.add_command(cli_metadata._toolchain_cli)
    root.add_command(cli_baseline._baseline_cli)
    root.add_command(cli_baseline._baseline_export_cli)
    root.add_command(cli_dataset._dataset_cli)
    root.add_command(cli_dataset._dataset_migrate_sol_cli)
    root.add_command(cli_dataset._dataset_migrate_flashinfer_cli)
    root.add_command(cli_evaluation_runtime._eval_runtime_cli)
```

- [ ] **Step 4: Update `main.py` minimally**

If `commands.py` was created, replace direct subcommand registration imports with:

```python
from .commands import register_subcommands
```

and call:

```python
register_subcommands(cli)
```

at the same location where the commands are currently added.

- [ ] **Step 5: Lower the fan-out budget only after imports move**

If `cli.main` internal imports dropped, update:

```python
assert len(main_imports) <= 17
```

to the new observed count. Do not lower it below the observed count.

- [ ] **Step 6: Run focused CLI tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py tests/sol_execbench/test_cli_metadata_commands.py tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_cli_baseline_commands.py tests/sol_execbench/test_cli_evaluation_runtime.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/sol_execbench/cli/main.py src/sol_execbench/cli/commands.py tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Bound CLI main coupling"
```

If `commands.py` was not created, omit it from `git add`.

---

### Task 3: Stop Internal Callers From Depending On Dataset Package Re-exports

**Files:**
- Modify: internal callers found by search
- Modify only if safe: `src/sol_execbench/core/dataset/__init__.py`
- Test: relevant dataset and CLI tests

- [ ] **Step 1: Find internal imports from the dataset package root**

Run:

```bash
rg "from sol_execbench\\.core\\.dataset import|import sol_execbench\\.core\\.dataset" src tests scripts
```

Expected: A finite list of call sites. Separate internal source call sites under `src/` from tests and scripts.

- [ ] **Step 2: Replace internal source imports with direct module imports**

For each `src/` call site that imports from `sol_execbench.core.dataset`, replace the package-root import with the defining module. Examples:

```python
from sol_execbench.core.dataset.manifest import DatasetManifest
from sol_execbench.core.dataset.readiness import DatasetReadiness
from sol_execbench.core.dataset.sharding import DatasetShardPlan
```

Keep tests and public scripts on package-root imports unless the import is clearly internal-only.

- [ ] **Step 3: Run dataset-focused tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_dataset_runner.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_dataset_solutions.py -q
```

Expected: PASS.

- [ ] **Step 4: Trim `dataset.__init__` only when no public import breaks**

If Step 1 shows names that are not imported by tests, docs examples, or scripts, remove only those unused re-exports from `src/sol_execbench/core/dataset/__init__.py`.

After each removal, run:

```bash
uv run pytest tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_dataset_contract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sol_execbench/core/dataset tests/sol_execbench
git commit -s -m "#0 - Narrow dataset package imports"
```

---

### Task 4: Stop Internal Callers From Depending On Scoring Package Re-exports

**Files:**
- Modify: internal callers found by search
- Modify only if safe: `src/sol_execbench/core/scoring/__init__.py`
- Test: scoring and report tests

- [ ] **Step 1: Find internal imports from the scoring package root**

Run:

```bash
rg "from sol_execbench\\.core\\.scoring import|import sol_execbench\\.core\\.scoring" src tests scripts
```

Expected: A finite list of call sites. Treat public scripts and tests as compatibility signals.

- [ ] **Step 2: Replace internal source imports with direct module imports**

For each `src/` call site that imports from `sol_execbench.core.scoring`, replace the package-root import with the defining module. Examples:

```python
from sol_execbench.core.scoring.amd_score import AmdNativeScore
from sol_execbench.core.scoring.amd_sol_v2 import AmdSolBoundV2Artifact
from sol_execbench.core.scoring.official_score import OfficialScoreEvidence
```

- [ ] **Step 3: Run scoring-focused tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_official_score_evidence.py -q
```

Expected: PASS.

- [ ] **Step 4: Trim `scoring.__init__` only when no compatibility import breaks**

Remove unused re-exports from `src/sol_execbench/core/scoring/__init__.py` one group at a time. After each group, run:

```bash
uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_official_score_evidence.py tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sol_execbench/core/scoring tests/sol_execbench
git commit -s -m "#0 - Narrow scoring package imports"
```

---

### Task 5: Protect High-Inbound Model Files From New Runtime Dependencies

**Files:**
- Modify: `tests/sol_execbench/test_cli_module_boundaries.py`

- [ ] **Step 1: Add explicit leaf-model dependency test**

Append this test to `tests/sol_execbench/test_cli_module_boundaries.py`:

```python
def test_high_inbound_model_modules_stay_lightweight() -> None:
    allowed_targets = {
        "sol_execbench.core.data.base_model",
        "sol_execbench.core.scoring.amd_bound_graph_models",
        "sol_execbench.core.scoring.amd_hardware_models",
        "sol_execbench.core.scoring.solar_derivation_models",
    }
    maximum_internal_imports = {
        "sol_execbench.core.data.base_model": 0,
        "sol_execbench.core.scoring.amd_bound_graph_models": 1,
        "sol_execbench.core.scoring.amd_hardware_models": 0,
        "sol_execbench.core.scoring.solar_derivation_models": 2,
    }
    edges = _internal_import_edges()

    observed = {
        module: sorted(target for source, target in edges if source == module)
        for module in allowed_targets
    }

    assert {
        module: len(targets) for module, targets in observed.items()
    } == maximum_internal_imports
```

- [ ] **Step 2: Run boundary tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/sol_execbench/test_cli_module_boundaries.py
git commit -s -m "#0 - Protect foundational model imports"
```

---

### Task 6: Final Verification

**Files:**
- No planned source edits

- [ ] **Step 1: Run focused boundary and coupling tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_module_boundaries.py -q
```

Expected: PASS.

- [ ] **Step 2: Run relevant functional suites**

Run:

```bash
uv run pytest tests/sol_execbench/test_cli_metadata_commands.py tests/sol_execbench/test_cli_dataset_commands.py tests/sol_execbench/test_cli_baseline_commands.py tests/sol_execbench/test_cli_evaluation_runtime.py tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_official_score_evidence.py -q
```

Expected: PASS.

- [ ] **Step 3: Run lint**

Run:

```bash
uv run --with ruff ruff check .
```

Expected: PASS.

- [ ] **Step 4: Optional full suite**

Run when local time and ROCm environment allow:

```bash
uv run pytest tests/ -q
```

Expected: PASS or documented skips for hardware-marked tests.

- [ ] **Step 5: Commit any verification-only doc updates**

If no files changed during verification, do not commit. If documentation was updated, run:

```bash
git add docs/superpowers/plans/2026-07-08-coupling-hardening.md
git commit -s -m "#0 - Document coupling hardening plan"
```

---

## Execution Notes

- Do not change public data schemas in `core.data`, `core.dataset.manifest`, or scoring model files unless a test explicitly requires it.
- Do not remove package-root re-exports just because internal callers no longer need them; tests, scripts, and docs are compatibility signals.
- Lower coupling budgets only after observing the new import graph. Budget tests should prevent regression, not force arbitrary churn.
- Keep each task in a separate signed commit so regressions can be bisected cleanly.
