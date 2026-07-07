# AMD Score Report Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract AMD score report construction from `src/sol_execbench/core/dataset/runner.py` into a focused module without changing existing callers.

**Architecture:** Add `src/sol_execbench/core/dataset/amd_score_reports.py` to own AMD score report parsing, generation, and writing. Keep `runner.py` as the compatibility surface by wrapping the new implementation and passing the active `runner.run_cli` dependency through the wrapper so monkeypatch behavior remains intact.

**Tech Stack:** Python 3.12, pytest, Pydantic project models, existing SOL ExecBench dataset/scoring modules.

---

## File Structure

- Create `src/sol_execbench/core/dataset/amd_score_reports.py`: extracted AMD score report implementation and helper parsing functions.
- Modify `src/sol_execbench/core/dataset/runner.py`: remove extracted helper implementations, import the new module, and keep compatibility wrappers for `build_amd_score_reports_for_problem` and `write_amd_score_report`.
- Modify `tests/sol_execbench/test_run_dataset_amd_score.py`: add a focused compatibility test proving `runner.run_cli` monkeypatching still affects score report construction after extraction.

## Task 1: Lock Runner Monkeypatch Compatibility

**Files:**
- Modify: `tests/sol_execbench/test_run_dataset_amd_score.py`

- [ ] **Step 1: Write the failing test**

Add this import near the existing imports if `pytest` is not already imported in the file:

```python
import pytest
```

Add this test near the existing tests that monkeypatch `run_dataset.run_cli` or `build_amd_score_reports_for_problem.__globals__`:

```python
def test_runner_score_report_wrapper_uses_runner_run_cli(monkeypatch):
    from sol_execbench.core.dataset import runner

    def fake_run_cli(*args, **kwargs):
        return [{"evaluation": {"environment": {"hardware": "gfx1200"}}}]

    monkeypatch.setattr(runner, "run_cli", fake_run_cli)

    assert runner.build_amd_score_reports_for_problem.__globals__["run_cli"] is fake_run_cli
```

This test intentionally checks the current compatibility contract used by existing tests: callers can patch `runner.run_cli`, and the public runner function still resolves that active global.

- [ ] **Step 2: Run test to verify it fails after extraction-only implementation**

Run:

```bash
uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py::test_runner_score_report_wrapper_uses_runner_run_cli -v
```

Expected before implementation:

```text
FAILED ... KeyError: 'run_cli'
```

If the test passes before any production changes, tighten it by asserting the wrapper is the runner-level public function:

```python
assert runner.build_amd_score_reports_for_problem.__module__ == "sol_execbench.core.dataset.runner"
assert runner.build_amd_score_reports_for_problem.__globals__["run_cli"] is fake_run_cli
```

- [ ] **Step 3: Commit the failing test**

Do not commit if the test does not fail for the expected compatibility reason.

```bash
git add tests/sol_execbench/test_run_dataset_amd_score.py
git commit -s -m "#0 - Add AMD score report compatibility test"
```

## Task 2: Extract AMD Score Report Module

**Files:**
- Create: `src/sol_execbench/core/dataset/amd_score_reports.py`
- Modify: `src/sol_execbench/core/dataset/runner.py`
- Test: `tests/sol_execbench/test_run_dataset_amd_score.py`

- [ ] **Step 1: Create the new module with extracted imports and helpers**

Create `src/sol_execbench/core/dataset/amd_score_reports.py` with the extracted implementation. Use this structure:

```python
"""AMD-native score report helpers for dataset-scale runs."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.dataset.evidence_refs import sidecar_stem_for_workload
from sol_execbench.core.scoring.amd_hardware_models import (
    EstimateConfidence,
    amd_hardware_model_from_dict,
)
from sol_execbench.core.scoring.amd_score import (
    AmdNativeScore,
    build_amd_native_suite_report,
    score_amd_native_trace_workload,
)
from sol_execbench.core.scoring.amd_sol import default_amd_hardware_models
from sol_execbench.core.scoring.amd_sol_v2 import (
    AMD_SOL_V2_SCHEMA_VERSION,
    AmdSolBoundV2Artifact,
    AmdSolV2AggregateBound,
    AmdSolV2CoverageSummary,
    build_amd_sol_bound_v2_artifact,
)
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact
from sol_execbench.core.scoring.solar_derivation import (
    SolarAggregateStatus,
    build_solar_derivation_evidence,
    solar_derivation_from_dict,
)

RunCliFunc = Callable[..., list[dict]]
```

Move these functions from `runner.py` unchanged unless the step below says otherwise:

```python
def _hardware_model_key_from_trace_payloads(traces_payload: Sequence[dict]) -> str:
    ...

def _read_json_object(path: Path) -> dict | None:
    ...

def _minimal_amd_sol_bound_v2_from_payload(payload: dict) -> AmdSolBoundV2Artifact | None:
    ...

def _minimal_solar_aggregate_from_payload(payload: dict) -> SolarAggregateStatus | None:
    ...

def write_amd_score_report(
    report_path: Path,
    amd_scores: list[AmdNativeScore],
    *,
    problem_count: int,
    baseline_entry_count: int,
) -> None:
    ...
```

- [ ] **Step 2: Move score construction into an implementation function**

In `amd_score_reports.py`, name the extracted score construction function `_build_amd_score_reports_for_problem_impl` and add a required keyword-only `run_cli_func` parameter, even if the current implementation only needs it for compatibility:

```python
def _build_amd_score_reports_for_problem_impl(
    *,
    definition_payload: dict,
    workload_path: Path,
    traces_payload: list[dict],
    trace_ref: str,
    run_cli_func: RunCliFunc,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    sol_bound_artifact_dir: Path | None = None,
    solar_derivation_dir: Path | None = None,
    sidecar_namespace: str | None = None,
    derived_sidecar_exclusions: dict[str, str] | None = None,
) -> list[AmdNativeScore]:
    """Build derived AMD-native scores for one dataset-run problem."""
    _ = run_cli_func
    definition = Definition(**definition_payload)
    workloads = {
        workload.uuid: workload
        for workload in (
            Workload(**json.loads(line))
            for line in workload_path.read_text().splitlines()
            if line.strip()
        )
    }
    hardware_models = default_amd_hardware_models()
    hardware_model_key = _hardware_model_key_from_trace_payloads(traces_payload)
    hardware_model = hardware_models[hardware_model_key]
    scores: list[AmdNativeScore] = []
    derived_sidecar_exclusions = derived_sidecar_exclusions or {}

    for trace_index, trace_payload in enumerate(traces_payload):
        trace = Trace(**trace_payload)
        workload = workloads.get(trace.workload.uuid)
        derived_exclusion = derived_sidecar_exclusions.get(trace.workload.uuid)
        sidecar_stem = (
            sidecar_stem_for_workload(
                definition.name,
                trace.workload.uuid,
                problem_namespace=sidecar_namespace,
            )
            if workload is not None
            else None
        )
        sol_bound_ref = (
            f"derived:{definition.name}:{trace.workload.uuid}:amd_sol_bound_v2"
        )
        sol_bound_path = (
            sol_bound_artifact_dir / f"{sidecar_stem}.amd-sol-v2.json"
            if sol_bound_artifact_dir is not None and sidecar_stem is not None
            else None
        )
        artifact = None
        if sol_bound_path is not None and sol_bound_path.exists():
            existing_payload = _read_json_object(sol_bound_path)
            if existing_payload is not None:
                artifact = _minimal_amd_sol_bound_v2_from_payload(existing_payload)
                if artifact is not None:
                    sol_bound_ref = str(sol_bound_path)
        if artifact is None and workload is not None and derived_exclusion is None:
            artifact = build_amd_sol_bound_v2_artifact(
                definition,
                workload,
                hardware_model,
                hardware_model_ref=f"default_amd_hardware_models.{hardware_model_key}",
            )
        solar_derivation = None
        derived_evidence_refs = None
        solar_derivation_ref = (
            f"derived:{definition.name}:{trace.workload.uuid}:solar_derivation"
        )
        solar_derivation_path = (
            solar_derivation_dir / f"{sidecar_stem}.solar-derivation.json"
            if solar_derivation_dir is not None and sidecar_stem is not None
            else None
        )
        if solar_derivation_path is not None and solar_derivation_path.exists():
            existing_payload = _read_json_object(solar_derivation_path)
            if existing_payload is not None:
                solar_derivation = _minimal_solar_aggregate_from_payload(
                    existing_payload
                )
                if solar_derivation is not None:
                    solar_derivation_ref = str(solar_derivation_path)
        if (
            workload is not None
            and solar_derivation_dir is not None
            and derived_exclusion is None
        ):
            solar_derivation_dir.mkdir(parents=True, exist_ok=True)
            assert solar_derivation_path is not None
            if solar_derivation is None:
                generated = build_solar_derivation_evidence(definition, workload)
                generated_payload = generated.to_dict()
                solar_derivation_path.write_text(
                    json.dumps(generated_payload, indent=2)
                )
                solar_derivation_ref = str(solar_derivation_path)
                try:
                    solar_derivation = solar_derivation_from_dict(generated_payload)
                except ValueError as exc:
                    solar_derivation = None
                    derived_evidence_refs = {"solar_derivation_parse_error": str(exc)}
            else:
                solar_derivation_ref = str(solar_derivation_path)
            derived_evidence_refs = {
                "formula": f"{solar_derivation_ref}#groups.formula_evidence",
                "hardware_model": f"default_amd_hardware_models.{hardware_model_key}",
                "coverage": f"{solar_derivation_ref}#coverage_summary",
                "score_eligibility": f"{solar_derivation_ref}#aggregate_status",
                **(derived_evidence_refs or {}),
            }
        elif derived_exclusion is not None:
            derived_evidence_refs = {
                "derived_sidecar_exclusion": derived_exclusion,
                "hardware_model": f"default_amd_hardware_models.{hardware_model_key}",
            }
        if (
            artifact is not None
            and sol_bound_path is not None
            and sol_bound_artifact_dir is not None
        ):
            sol_bound_artifact_dir.mkdir(parents=True, exist_ok=True)
            if not sol_bound_path.exists():
                sol_bound_path.write_text(json.dumps(artifact.to_dict(), indent=2))
            sol_bound_ref = str(sol_bound_path)
        scores.append(
            score_amd_native_trace_workload(
                trace,
                artifact,
                trace_ref=trace_ref,
                timing_evidence_ref=trace_ref,
                sol_bound_ref=sol_bound_ref,
                baseline_ref=(
                    f"{baseline_artifact.source}#{definition.name}:{trace.workload.uuid}"
                    if baseline_artifact
                    and baseline_artifact.lookup(definition.name, trace.workload.uuid)
                    is not None
                    else "trace.evaluation.performance.reference_latency_ms"
                ),
                baseline_artifact=baseline_artifact,
                hardware_model_ref=f"default_amd_hardware_models.{hardware_model_key}",
                solar_derivation=solar_derivation,
                derived_evidence_refs=derived_evidence_refs,
            )
        )
        if trace_index % 16 == 0:
            gc.collect()
    return scores
```

- [ ] **Step 3: Add runner compatibility wrappers**

In `src/sol_execbench/core/dataset/runner.py`, replace the moved implementation with imports and wrappers:

```python
from sol_execbench.core.dataset.amd_score_reports import (
    _build_amd_score_reports_for_problem_impl,
    write_amd_score_report,
)
```

Keep this wrapper in `runner.py`:

```python
def build_amd_score_reports_for_problem(
    *,
    definition_payload: dict,
    workload_path: Path,
    traces_payload: list[dict],
    trace_ref: str,
    baseline_artifact: ScoringBaselineArtifact | None = None,
    sol_bound_artifact_dir: Path | None = None,
    solar_derivation_dir: Path | None = None,
    sidecar_namespace: str | None = None,
    derived_sidecar_exclusions: dict[str, str] | None = None,
) -> list[AmdNativeScore]:
    """Build derived AMD-native scores for one dataset-run problem."""
    return _build_amd_score_reports_for_problem_impl(
        definition_payload=definition_payload,
        workload_path=workload_path,
        traces_payload=traces_payload,
        trace_ref=trace_ref,
        run_cli_func=run_cli,
        baseline_artifact=baseline_artifact,
        sol_bound_artifact_dir=sol_bound_artifact_dir,
        solar_derivation_dir=solar_derivation_dir,
        sidecar_namespace=sidecar_namespace,
        derived_sidecar_exclusions=derived_sidecar_exclusions,
    )
```

Remove now-unused imports from `runner.py`:

```python
from collections.abc import Sequence
from sol_execbench.core.dataset.evidence_refs import sidecar_stem_for_workload
from sol_execbench.core.scoring.amd_score import build_amd_native_suite_report
from sol_execbench.core.scoring.amd_score import score_amd_native_trace_workload
from sol_execbench.core.scoring.amd_sol import default_amd_hardware_models
from sol_execbench.core.scoring.amd_sol_v2 import AMD_SOL_V2_SCHEMA_VERSION
from sol_execbench.core.scoring.amd_sol_v2 import AmdSolBoundV2Artifact
from sol_execbench.core.scoring.amd_sol_v2 import AmdSolV2AggregateBound
from sol_execbench.core.scoring.amd_sol_v2 import AmdSolV2CoverageSummary
from sol_execbench.core.scoring.amd_sol_v2 import build_amd_sol_bound_v2_artifact
from sol_execbench.core.scoring.amd_hardware_models import EstimateConfidence
from sol_execbench.core.scoring.amd_hardware_models import amd_hardware_model_from_dict
from sol_execbench.core.scoring.solar_derivation import SolarAggregateStatus
from sol_execbench.core.scoring.solar_derivation import build_solar_derivation_evidence
from sol_execbench.core.scoring.solar_derivation import solar_derivation_from_dict
```

Keep these imports because `runner.py` still uses them in wrappers or other functions:

```python
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_score import AmdNativeScore
from sol_execbench.core.scoring.baseline_artifact import ScoringBaselineArtifact
```

- [ ] **Step 4: Run focused compatibility test**

Run:

```bash
uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py::test_runner_score_report_wrapper_uses_runner_run_cli -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Run AMD score report regression tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_run_dataset_amd_score.py
```

Expected:

```text
... passed
```

- [ ] **Step 6: Run dataset runner regression tests**

Run:

```bash
uv run pytest tests/sol_execbench/test_dataset_runner.py
```

Expected:

```text
... passed
```

- [ ] **Step 7: Run import/lint sanity checks**

Run:

```bash
uv run --with ruff ruff check src/sol_execbench/core/dataset/runner.py src/sol_execbench/core/dataset/amd_score_reports.py tests/sol_execbench/test_run_dataset_amd_score.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 8: Commit implementation**

```bash
git add src/sol_execbench/core/dataset/runner.py src/sol_execbench/core/dataset/amd_score_reports.py tests/sol_execbench/test_run_dataset_amd_score.py
git commit -s -m "#0 - Extract AMD score report helpers"
```

## Self-Review

- Spec coverage: Task 1 covers compatibility; Task 2 covers module extraction, runner wrappers, output behavior regressions, and no cycle by one-way import.
- Placeholder scan: No placeholder steps; each code-changing step includes concrete code or exact move instructions.
- Type consistency: The wrapper keeps the existing `build_amd_score_reports_for_problem` signature and returns `list[AmdNativeScore]`; the new implementation uses `RunCliFunc = Callable[..., list[dict]]`.
