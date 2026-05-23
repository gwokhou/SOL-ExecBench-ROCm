# Phase 55: Ready Subset Selection And Bounded Execution Closure - Research

**Researched:** 2026-05-23
**Domain:** Python dataset execution orchestration, sidecar report generation, ROCm benchmark evidence closure
**Confidence:** HIGH

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Extend `scripts/run_dataset.py` to read Phase 54 `ready_subset.json` and select only referenced problem/workload pairs while preserving the existing `sol-execbench` subprocess path.
- Generate temporary filtered workload files only under output or execution staging paths. Never modify canonical dataset `workload.jsonl`.
- If no ready workloads are selected, generate a closure report with `no_ready_workloads` semantics and exit successfully.
- Apply filters by intersection: ready subset, category, limit, workload cap, and runner filters all contribute visible filtered/not-attempted reasons.
- Reuse or extend existing `run_dataset.py` controls for category, limit, workload cap, timeout, warmup, iterations, rerun policy, and derived evidence flags.
- Closure provenance should record command args, dataset manifest checksum, readiness/ready-subset checksum, git commit, solution mode/name, benchmark config, timestamp, and output paths.
- `skipped_existing_pass` is a first-class closure state and should still generate or reference derived sidecars unless rerun policy requires execution.
- Per-workload/problem failures should not prevent the closure report from being written; report the failure state explicitly.
- Emit `execution_closure.json` and optionally a lightweight Markdown summary. Full parity gap reporting remains Phase 56.
- Use closure statuses `not_attempted`, `filtered`, `skipped_existing_pass`, `attempted_passed`, `attempted_failed`, `missing_trace`, and `derived_evidence_missing`.
- Join readiness, traces, summaries, logs, AMD score, AMD SOL v2, SOLAR derivation, and timing sidecar paths by problem ID plus workload UUID or row index.
- Read and reference canonical trace JSONL only; do not add closure fields or rewrite trace JSONL.
- Automated tests should use fixtures and monkeypatched runner functions rather than real ROCm/GPU execution.
- If GPU hardware exists, real sample execution is optional manual validation, not a phase-pass requirement.
- Derived evidence checks should reuse existing Phase 52 sidecar generation and reference behavior.
- Claim guardrails must say bounded ready-subset closure is not full 235 validation, paper parity, or leaderboard result; failures, blockers, and not-attempted states must stay visible.

### the agent's Discretion
The agent may choose exact helper names, closure schema fields, and Markdown summary format as long as existing runner semantics are preserved, outputs are deterministic, and canonical dataset/traces stay unchanged.

### Deferred Ideas (OUT OF SCOPE)
- Full parity gap report aggregation is deferred to Phase 56.
- Milestone release claim closure is deferred to Phase 57.
- Real full-suite hardware validation remains out of scope.

## Summary

Phase 55 should be implemented as a bounded selection and closure layer inside `scripts/run_dataset.py`, not as a second runner. The existing script already discovers problems, applies category/limit controls, creates temporary workload files for `--max-workloads`, skips existing passing traces unless `--rerun`, calls `sol-execbench` through `run_cli`, writes `traces.json` and `summary.json`, and generates AMD score, AMD SOL v2, SOLAR derivation, and timing sidecars. [VERIFIED: scripts/run_dataset.py]

The minimal implementation path is to add a `--ready-subset` input and `--execution-closure` output, load Phase 54 ready-subset/readiness metadata, intersect it with existing runner filters, materialize per-problem filtered workload files under the problem output directory, and append deterministic closure records as each problem is skipped, attempted, filtered, or fails. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] [VERIFIED: scripts/run_dataset.py]

**Primary recommendation:** Add small pure helpers for ready-subset loading, workload selection, filtered workload writing, trace/evidence indexing, and closure serialization, then thread them through the current `main()` loop so `run_cli()` remains the only benchmark execution path. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXEC-01 | Run a bounded ready subset through `scripts/run_dataset.py` and the primary `sol-execbench` subprocess path without a second runner. | Add `--ready-subset` to `run_dataset.py`; keep `run_cli()` as the execution seam. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: scripts/run_dataset.py] |
| EXEC-02 | Constrain execution by category, limit, workload cap, timeout, warmup, iterations, rerun policy, and derived-evidence flags. | Reuse existing `--category`, `--limit`, `--max-workloads`, `--timeout`, `--warmup-runs`, `--iterations`, `--rerun`, `--amd-score-report`, `--amd-sol-bound-dir`, `--solar-derivation`, and `--timing-evidence-dir`. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: scripts/run_dataset.py] |
| EXEC-03 | Join readiness results to canonical traces, `summary.json`, CLI logs, skipped-existing-pass states, missing trace states, failures, and not-attempted items. | Build closure records keyed by `problem_id` plus `workload_uuid` or `row_index`; include summary/log/trace refs and explicit status vocabulary. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: src/sol_execbench/core/dataset/readiness.py] [VERIFIED: scripts/run_dataset.py] |
| EXEC-04 | Reference AMD-native score reports, AMD SOL v2 sidecars, SOLAR derivation sidecars, and timing evidence without mutating trace JSONL. | Current derived reports are separate artifacts and public guardrails require trace JSONL to exclude derived evidence fields. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] [VERIFIED: docs/analysis.md] |
| EXEC-05 | Record command arguments, dataset manifest checksum, git commit, solution mode/name, and benchmark config provenance. | Add a provenance object to `execution_closure.json`; existing ready-subset has readiness checksum and its own checksum, and `run_dataset.py` builds benchmark config. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] [VERIFIED: scripts/run_dataset.py] |

## Project Constraints (from AGENTS.md)

- Python package source is under `src/sol_execbench/`, CLI entry point is `sol_execbench.cli:cli`, and helper scripts live under `scripts/`. [VERIFIED: AGENTS.md]
- Use Python 3.12+ and Ruff-enforced style; keep focused changes consistent with nearby modules and avoid broad refactors. [VERIFIED: AGENTS.md]
- Tests belong under `tests/sol_execbench/` for package behavior and use pytest. [VERIFIED: AGENTS.md]
- Environment-sensitive tests should use existing markers including `cpp`, `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and legacy `requires_cutile`. [VERIFIED: AGENTS.md] [VERIFIED: pyproject.toml]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, downloaded datasets, local cache, build output, or benchmark output. [VERIFIED: AGENTS.md]
- ROCm >= 7.0 is the supported software baseline, and RDNA 4 plus CDNA 3 remain project targets. [VERIFIED: AGENTS.md]
- Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable. [VERIFIED: AGENTS.md]
- NVIDIA/CUDA paths may be removed rather than maintained as a dual backend. [VERIFIED: AGENTS.md]
- Direct repo edits should happen through GSD workflow unless explicitly bypassed. [VERIFIED: AGENTS.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Ready-subset selection | Script orchestration (`scripts/run_dataset.py`) | Dataset sidecar models | The runner owns execution filters; Phase 54 models own ready-subset shape. [VERIFIED: scripts/run_dataset.py] [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] |
| Temporary workload files | Script orchestration output/staging paths | Canonical dataset files are read-only inputs | Existing `--max-workloads` already writes a temporary `workload.jsonl` under each problem output directory. [VERIFIED: scripts/run_dataset.py] |
| Benchmark execution | Primary CLI subprocess | Benchmark config sidecar | `run_cli()` builds and invokes `sol-execbench` and parses JSON traces. [VERIFIED: scripts/run_dataset.py] |
| Closure report | Script orchestration sidecar | Derived score/timing sidecars | Closure is a run-level artifact that joins existing outputs without changing trace schema. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| Derived evidence references | Existing score/timing helpers | Closure report | AMD score, AMD SOL v2, SOLAR derivation, and timing evidence are already emitted as separate files. [VERIFIED: scripts/run_dataset.py] [VERIFIED: docs/analysis.md] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12.13 available locally | Runner orchestration and tests | Project requires Python 3.12+ and the local uv environment reports Python 3.12.13. [VERIFIED: pyproject.toml] [VERIFIED: uv run python --version] |
| stdlib `argparse`, `json`, `pathlib`, `subprocess`, `shutil`, `hashlib` | Python stdlib | CLI flags, sidecar JSON, paths, subprocess execution, checksums | `run_dataset.py` already uses these modules for the relevant behavior. [VERIFIED: scripts/run_dataset.py] |
| Pydantic v2 | `>=2.12.5` configured | Existing dataset/readiness models | `DatasetReadiness` and `ReadySubset` are Pydantic models and pyproject pins Pydantic v2-compatible dependency. [VERIFIED: src/sol_execbench/core/dataset/readiness.py] [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] [VERIFIED: pyproject.toml] |
| pytest | 9.0.2 available locally | Unit and integration-style fixture tests | Project config uses pytest and the local environment reports pytest 9.0.2. [VERIFIED: pyproject.toml] [VERIFIED: uv run pytest --version] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Click/Rich | Click `>=8.0`, Rich `>=13.0` configured | Primary CLI package dependencies | Do not add Phase 55 options to primary `sol-execbench`; public guardrails check that derived workflow options stay off primary CLI. [VERIFIED: pyproject.toml] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| torch ROCm / triton-rocm | Configured in pyproject | Real benchmark execution when available | Tests for this phase should monkeypatch `run_cli()` and avoid requiring real ROCm hardware. [VERIFIED: pyproject.toml] [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Extending `run_dataset.py` | New closure runner script | Rejected by EXEC-01 and context because it would introduce a second benchmark runner. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] |
| Pydantic closure models | Plain dict assembly only | Pydantic is optional; plain dicts match current `run_dataset.py` style, but a small model improves status vocabulary validation. [VERIFIED: scripts/run_dataset.py] [ASSUMED] |
| Rewriting canonical trace JSONL | Add closure fields to traces | Rejected by public contract guardrails and phase context. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] |

**Installation:** No new external package installation is recommended. [VERIFIED: pyproject.toml]

## Package Legitimacy Audit

No new external packages are recommended for Phase 55, so the package legitimacy gate is not applicable. [VERIFIED: pyproject.toml]

## Run Dataset Integration Points

Add script options only to `scripts/run_dataset.py`: `--ready-subset PATH`, `--execution-closure PATH`, optional `--execution-closure-summary PATH`, and optional `--dataset-manifest PATH` for provenance checksum loading. [VERIFIED: scripts/run_dataset.py] [VERIFIED: .planning/REQUIREMENTS.md]

Thread selection after `discover_problems()` and before the per-problem loop. `discover_problems()` already returns category/problem paths, and `--limit` currently slices the problem list before execution. [VERIFIED: scripts/run_dataset.py]

Keep `run_cli()` unchanged as the benchmark subprocess boundary. `build_cli_command()` constructs the `sol-execbench` command with definition, workload, solution, timeout, config, staging, verbosity, and `--json`. [VERIFIED: scripts/run_dataset.py]

Keep skipped-existing-pass behavior, but emit closure records for every workload represented by the existing passing trace. Existing tests already assert derived reports and SOLAR sidecars are generated when passing traces are skipped. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

Preserve existing failure handling: a per-problem `run_cli()` failure should append a failed summary and continue to final `summary.json`; Phase 55 should also append `attempted_failed` or `missing_trace` closure records and still write `execution_closure.json`. [VERIFIED: scripts/run_dataset.py] [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]

## Ready-Subset Filtering Strategy

Use ready subset problem refs as the first candidate universe, keyed by `problem_id` with workload refs keyed by `uuid` when present and `row_index` as fallback. `ReadySubsetProblemRef` stores `category`, `problem_id`, `problem_path`, and `workloads`; `ReadySubsetWorkloadRef` stores `uuid` and `row_index`. [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py]

Apply filters by intersection in this order for deterministic reporting: ready-subset membership, category filter, discovered problem existence, problem limit, per-problem `--max-workloads`, and solution/reference availability. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] [VERIFIED: scripts/run_dataset.py]

Closure should retain excluded ready-subset items as `filtered` with `filter_reasons`, and retain readiness workloads not in the ready subset as `not_attempted` when readiness metadata is available. This keeps blockers visible and prevents bounded execution from looking like complete validation. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] [VERIFIED: .planning/REQUIREMENTS.md]

If `ready_subset.problems` is empty or all entries are filtered out before execution, write `execution_closure.json` with top-level status `no_ready_workloads`, zero attempted counts, visible filter summaries, and exit code 0. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]

## Temporary Workload File Strategy

Never edit the canonical `problem_dir / "workload.jsonl"`. Existing code already writes a truncated workload to `problem_output_dir / "workload.jsonl"` when `--max-workloads` is set, then passes that temporary path to `run_cli()`. [VERIFIED: scripts/run_dataset.py]

For ready-subset filtering, write a filtered `workload.jsonl` under `output_dir / category / problem_name / "workload.jsonl"` before solution construction and CLI invocation. Use canonical workload rows selected by UUID or row index, preserving original row order. [VERIFIED: scripts/run_dataset.py] [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py]

If both ready-subset filtering and `--max-workloads` apply, first select ready-subset rows, then cap the selected rows; report rows beyond the cap as `filtered` with reason `max_workloads_cap`. This matches intersection semantics and keeps cap effects auditable. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] [VERIFIED: scripts/run_dataset.py]

Use output-relative references in closure when possible, matching `_extend_derived_reports_for_problem()` behavior for `trace_ref`. [VERIFIED: scripts/run_dataset.py]

## Closure Report Schema And Status Vocabulary

Recommended top-level JSON shape:

```json
{
  "schema_version": "sol_execbench.execution_closure.v1",
  "created_at": "2026-05-23T00:00:00Z",
  "status": "completed_with_failures",
  "provenance": {},
  "totals": {},
  "filters": {},
  "records": [],
  "claim_boundary": {}
}
```

Use top-level `status` values `completed`, `completed_with_failures`, and `no_ready_workloads`. These summarize the report only; per-workload closure state uses the locked vocabulary from context. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] [ASSUMED]

Use per-workload `closure_status` values exactly: `not_attempted`, `filtered`, `skipped_existing_pass`, `attempted_passed`, `attempted_failed`, `missing_trace`, and `derived_evidence_missing`. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]

Recommended record fields: `category`, `problem_id`, `problem_path`, `workload_uuid`, `row_index`, `readiness_status`, `readiness_reason_codes`, `closure_status`, `filter_reasons`, `trace_ref`, `summary_ref`, `cli_log_ref`, `amd_score_ref`, `amd_sol_bound_ref`, `solar_derivation_ref`, `timing_evidence_ref`, `solution_ref`, and `notes`. [VERIFIED: src/sol_execbench/core/dataset/readiness.py] [VERIFIED: scripts/run_dataset.py] [ASSUMED]

`derived_evidence_missing` should be used when an attempted/skipped workload has a canonical trace but a requested derived artifact is absent or unresolvable. Do not replace pass/fail status with this; include both trace outcome and evidence gap fields so Phase 56 can aggregate evidence completeness. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: docs/analysis.md] [ASSUMED]

## Provenance Fields

Record the exact `sys.argv`-style command args, resolved dataset root, selected categories, limit, max workloads, timeout, warmup runs, iterations, lock-clocks flag, rerun flag, keep-staging flag, verbose flag, solution mode (`reference`, `named_json`, `named_py`, or `missing`), solution name, output directory, summary path, derived report paths, and timing evidence root. [VERIFIED: scripts/run_dataset.py] [VERIFIED: .planning/REQUIREMENTS.md]

Record `ready_subset_checksum` and `readiness_checksum` from the ready-subset file. `ReadySubset` stores both fields when generated from readiness. [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py]

Record `dataset_manifest_checksum` if a dataset manifest path is supplied. Dataset manifest checksums exist in Phase 53 artifacts and are required by EXEC-05. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: tests/sol_execbench/test_dataset_contract.py]

Record `git_commit` with the current commit SHA when available; use `null` plus a warning when unavailable so local tarball runs still produce closure. [VERIFIED: .planning/REQUIREMENTS.md] [ASSUMED]

Record benchmark config both inline and by `config_path` when non-default timing settings cause `run_dataset.py` to write `output_dir / "config.json"`. [VERIFIED: scripts/run_dataset.py]

## Derived Evidence References

AMD-native score report references live at the user-provided `--amd-score-report` path, and per-score evidence refs include trace, timing, sol bound, baseline, and hardware model keys. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

AMD SOL v2 sidecars are generated under `--amd-sol-bound-dir` with stems from `_safe_sidecar_stem(definition.name, trace.workload.uuid)` and suffix `.amd-sol-v2.json`. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

SOLAR derivation sidecars are generated under `--solar-derivation` with the same safe stem and suffix `.solar-derivation.json`, and derived score refs point to formula, coverage, and score-eligibility anchors inside the sidecar. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

Timing evidence is written under the timing evidence root by category, with per-problem JSON named `{problem_output_dir.name}.timing.json`. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

Closure should derive evidence paths from actual file existence after helper calls, not by assuming requested options always produced files. Missing requested files should become `derived_evidence_missing` evidence gaps. [VERIFIED: .planning/REQUIREMENTS.md] [ASSUMED]

## Architecture Patterns

### System Architecture Diagram

```text
ready_subset.json + optional readiness/manifest
        |
        v
run_dataset.py argument parsing and provenance capture
        |
        v
discover_problems(dataset root, category filters)
        |
        v
intersect ready subset + category + limit + workload cap
        |
        +--> no selected workloads -> execution_closure.json(status=no_ready_workloads)
        |
        v
write filtered workload.jsonl under output/category/problem/
        |
        v
existing solution construction -> existing run_cli(sol-execbench --json)
        |
        v
canonical traces.json + summary.json + optional CLI logs
        |
        v
existing derived evidence helpers (AMD score, AMD SOL v2, SOLAR, timing)
        |
        v
execution_closure.json joins readiness + traces + summaries + evidence refs
```

### Recommended Project Structure

```text
scripts/run_dataset.py
  # add CLI options, selection helpers, closure aggregation, provenance capture

tests/sol_execbench/test_run_dataset_execution_closure.py
  # new focused fixture tests with monkeypatched run_cli

tests/sol_execbench/test_public_contract_guardrails.py
  # add closure sidecar-only and wording guardrails

docs/analysis.md
  # document bounded ready-subset closure command and claim boundary
```

### Pattern 1: Keep Helper Functions Pure

**What:** Implement selection and closure helpers as pure functions that accept parsed payloads and paths, returning dicts/lists for the caller to write. [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

**When to use:** Ready-subset row selection, path-safe evidence ref lookup, totals computation, and provenance assembly. [ASSUMED]

**Example:**

```python
def select_ready_workload_rows(
    workload_rows: list[dict],
    workload_refs: list[dict],
    *,
    max_workloads: int | None,
) -> tuple[list[dict], list[dict]]:
    """Return selected rows and filtered closure records."""
```

### Pattern 2: Monkeypatch Runner, Not GPU

**What:** Tests should import `scripts/run_dataset.py`, monkeypatch `run_cli`, set `sys.argv`, and call `run_dataset.main()`. [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

**When to use:** All Phase 55 automated execution tests, including pass, fail, skip, no-ready, and missing-derived-evidence cases. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]

### Anti-Patterns to Avoid

- **Adding a second execution script:** Violates EXEC-01 and risks diverging from `sol-execbench` subprocess semantics. [VERIFIED: .planning/REQUIREMENTS.md]
- **Adding closure options to primary `sol-execbench`:** Public guardrails require dataset/derived workflow options to stay off primary CLI help. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- **Mutating canonical workload or trace files:** Phase context and public guardrails require sidecar-only closure metadata. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- **Hiding filtered or not-attempted workloads:** Phase context requires visible filtered/not-attempted reasons. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Benchmark execution | A second subprocess runner | Existing `run_cli()` | It already builds the canonical `sol-execbench` command and parses JSON traces. [VERIFIED: scripts/run_dataset.py] |
| Dataset readiness classification | New readiness rules in Phase 55 | Phase 54 `readiness.json` and `ready_subset.json` | Readiness status and reason codes already exist and are deterministic sidecars. [VERIFIED: src/sol_execbench/core/dataset/readiness.py] [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] |
| Derived AMD evidence | New scoring/sidecar generation | Existing `_extend_derived_reports_for_problem()` and scoring helpers | Existing helpers generate AMD score refs, AMD SOL v2, and SOLAR derivation sidecars. [VERIFIED: scripts/run_dataset.py] |
| Path sanitization for sidecars | Ad hoc string concatenation | Existing `_safe_sidecar_stem()` for derived sidecars | Tests prove path-shaped identifiers stay inside requested sidecar dirs. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py] |

**Key insight:** Phase 55 is an audit join and bounded selection problem, not a benchmark semantics problem. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]

## Common Pitfalls

### Pitfall 1: Problem-Level Skips Hide Workload-Level Closure

**What goes wrong:** Existing skip logic skips a whole problem if all existing traces passed; closure still needs per-workload records. [VERIFIED: scripts/run_dataset.py]

**Why it happens:** `inspect_traces()` summarizes a problem, while ready subset and closure are workload-oriented. [VERIFIED: scripts/run_dataset.py] [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py]

**How to avoid:** Index trace payloads by workload UUID and row index fallback, then emit `skipped_existing_pass` per selected workload. [ASSUMED]

**Warning signs:** Closure totals count problems but not workloads, or skipped traces lack derived evidence refs. [ASSUMED]

### Pitfall 2: Limit Semantics Become Ambiguous

**What goes wrong:** Existing `--limit` slices discovered problems, but ready-subset filtering introduces another candidate universe. [VERIFIED: scripts/run_dataset.py]

**Why it happens:** `--limit` is a problem cap, not a workload cap. [VERIFIED: scripts/run_dataset.py]

**How to avoid:** Preserve `--limit` as a problem cap after ready-subset/category intersection and document `--max-workloads` as the per-problem workload cap. [VERIFIED: scripts/run_dataset.py] [ASSUMED]

**Warning signs:** Closure reports a workload cap as a problem limit, or limit filtering lacks explicit reason codes. [ASSUMED]

### Pitfall 3: Derived Evidence Assumptions Overclaim Completeness

**What goes wrong:** A trace exists but requested AMD SOL, SOLAR, AMD score, or timing artifacts are missing. [VERIFIED: .planning/REQUIREMENTS.md]

**Why it happens:** Existing helpers only generate derived artifacts when their flags are supplied and when enough data exists. [VERIFIED: scripts/run_dataset.py]

**How to avoid:** Check artifact existence after execution/skipping and record `derived_evidence_missing` gaps explicitly. [ASSUMED]

**Warning signs:** Closure says `attempted_passed` with no evidence refs despite derived-evidence flags. [ASSUMED]

### Pitfall 4: Report Wording Drifts Into Parity Claims

**What goes wrong:** A bounded run is described as full dataset validation or leaderboard-ready evidence. [VERIFIED: .planning/REQUIREMENTS.md]

**Why it happens:** Passing subset counts can be mistaken for full 235-problem coverage. [VERIFIED: .planning/REQUIREMENTS.md]

**How to avoid:** Include closure claim boundaries and add public contract tests/docs wording near existing v1.11 guardrails. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] [VERIFIED: docs/analysis.md]

**Warning signs:** Docs or reports say "validated", "paper parity", "leaderboard", or "full suite" without a negative boundary. [VERIFIED: .planning/REQUIREMENTS.md]

## Code Examples

### Runner Test Seam

```python
def run_cli(*args, **kwargs):
    return _matmul_trace_payload()

monkeypatch.setattr(run_dataset, "run_cli", run_cli)
monkeypatch.setattr(sys, "argv", ["run_dataset.py", str(dataset_root), "--ready-subset", str(subset_path)])
run_dataset.main()
```

Source: existing `test_dataset_runner_reruns_failed_existing_traces_before_reports`. [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py]

### Sidecar-Only Evidence Boundary

```python
for payload in (definition.model_dump(mode="json"), workload.model_dump(mode="json"), trace.model_dump(mode="json")):
    text = json.dumps(payload, sort_keys=True)
    assert "sol_execbench.ready_subset.v1" not in text
```

Source: existing v1.11 public contract guardrails. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct dataset batches by discovered problem only | Dataset sidecars now include manifest, inventory, readiness, and ready subset | v1.11 Phases 53-54 | Phase 55 should consume ready-subset sidecars instead of rediscovering readiness. [VERIFIED: .planning/ROADMAP.md] |
| Derived score data inside reports only | AMD score reports can reference AMD SOL v2 and SOLAR derivation sidecars | v1.10-v1.11 existing code | Closure can reference generated artifacts without changing trace schema. [VERIFIED: docs/analysis.md] [VERIFIED: scripts/run_dataset.py] |
| Hidden skip behavior | `skipped_existing_pass` must be first-class closure state | Phase 55 context | Closure must report skipped existing passing traces as evidence-bearing states. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] |

**Deprecated/outdated:**
- Treating `summary.json` as enough closure evidence is insufficient for EXEC-03 because it lacks readiness, filter, missing trace, and derived evidence joins. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: scripts/run_dataset.py]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A small Pydantic model may be useful for validating closure status vocabulary, but plain dicts also match local script style. | Standard Stack | Planner may choose a model or dicts; either is acceptable if tests enforce schema. |
| A2 | Top-level closure statuses should be `completed`, `completed_with_failures`, and `no_ready_workloads`. | Closure Report Schema And Status Vocabulary | User may prefer a different top-level vocabulary; per-workload vocabulary is locked. |
| A3 | Recommended record fields include refs and notes beyond the locked minimum. | Closure Report Schema And Status Vocabulary | Planner should keep the schema focused if implementation scope grows. |
| A4 | Git commit should be nullable with a warning outside git worktrees. | Provenance Fields | If release tooling requires a hard failure outside git, planner must adjust. |
| A5 | Artifact existence should be checked after helper calls rather than inferred from flags. | Derived Evidence References | If helpers return richer metadata later, direct returned refs may be better. |
| A6 | Workload UUID plus row index trace indexing is enough for Phase 55 fixture coverage. | Common Pitfalls | If real traces can duplicate UUIDs, row-index matching needs stronger validation. |

## Open Questions

1. **Should `--execution-closure` default to `output_dir / "execution_closure.json"` when `--ready-subset` is supplied?**
   - What we know: Phase context requires emitting `execution_closure.json`. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]
   - What's unclear: Whether users must pass an explicit output path. [ASSUMED]
   - Recommendation: Default to `output_dir / "execution_closure.json"` and allow override. [ASSUMED]

2. **Should readiness metadata be loaded separately or only through `ready_subset.json`?**
   - What we know: `ready_subset.json` has included ready workloads and checksums, but blocked workload details live in `readiness.json`. [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] [VERIFIED: src/sol_execbench/core/dataset/readiness.py]
   - What's unclear: Whether Phase 55 must emit not-attempted blockers for every non-ready workload before Phase 56. [ASSUMED]
   - Recommendation: Add optional `--readiness` to enrich not-attempted records; without it, report ready-subset filtered/not-attempted only. [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | Running tests and scripts | yes | executable at `/home/guohao/.cargo/bin/uv` | Use existing project commands through `uv`. [VERIFIED: which uv] |
| Python | Script/test runtime | yes | 3.12.13 | None needed. [VERIFIED: uv run python --version] |
| pytest | Automated tests | yes | 9.0.2 | None needed. [VERIFIED: uv run pytest --version] |
| ROCm GPU | Optional manual sample execution | not required for automated pass | not probed | Monkeypatch `run_cli()` in tests. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md] |

**Missing dependencies with no fallback:** None for automated Phase 55 planning. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]

**Missing dependencies with fallback:** Real ROCm hardware is optional manual validation; automated tests use fixtures and monkeypatched runner functions. [VERIFIED: .planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 [VERIFIED: uv run pytest --version] |
| Config file | `pyproject.toml` [VERIFIED: pyproject.toml] |
| Quick run command | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_public_contract_guardrails.py -q` [ASSUMED] |
| Full suite command | `uv run pytest tests/` [VERIFIED: AGENTS.md] |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| EXEC-01 | `--ready-subset` uses `run_cli()` and does not add a second runner | unit/integration fixture | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py::test_ready_subset_runs_through_existing_run_cli -q` | No, Wave 0 |
| EXEC-02 | Category, limit, max workloads, timeout/config, rerun, and derived flags all affect closure deterministically | unit/integration fixture | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py::test_ready_subset_filters_are_intersected_and_reported -q` | No, Wave 0 |
| EXEC-03 | Closure joins readiness, traces, summary, logs, skips, missing traces, failures, and not-attempted items | unit/integration fixture | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_records_all_statuses -q` | No, Wave 0 |
| EXEC-04 | Closure references AMD score, AMD SOL v2, SOLAR derivation, and timing evidence without mutating trace JSON | unit/integration fixture + contract | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_references_derived_evidence_sidecars -q` | No, Wave 0 |
| EXEC-05 | Closure provenance records command args, checksums, git commit, solution mode/name, and benchmark config | unit/integration fixture | `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py::test_execution_closure_records_provenance -q` | No, Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py -q` [ASSUMED]
- **Per wave merge:** `uv run pytest tests/sol_execbench/test_run_dataset_execution_closure.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -q` [ASSUMED]
- **Phase gate:** `uv run pytest tests/` before `$gsd-verify-work`. [VERIFIED: AGENTS.md]

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_run_dataset_execution_closure.py` — covers EXEC-01 through EXEC-05. [ASSUMED]
- [ ] Add public contract guardrail assertions for `sol_execbench.execution_closure.v1` staying sidecar-only and docs/report wording avoiding full validation claims. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] [ASSUMED]
- [ ] Add docs example in `docs/analysis.md` for `--ready-subset` plus `--execution-closure`. [VERIFIED: docs/analysis.md] [ASSUMED]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface in local script. [VERIFIED: scripts/run_dataset.py] |
| V3 Session Management | no | No session surface in local script. [VERIFIED: scripts/run_dataset.py] |
| V4 Access Control | no | Local filesystem paths only; validate sidecar path handling. [VERIFIED: scripts/run_dataset.py] |
| V5 Input Validation | yes | Validate ready-subset JSON structure with existing Pydantic models or explicit schema checks; reject unsafe path traversal for output-derived refs. [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py] |
| V6 Cryptography | no | Existing checksums are stable JSON checksums, not security signatures. [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] |

### Known Threat Patterns for This Phase

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal through problem/workload identifiers | Tampering | Use canonical discovered problem dirs and existing `_safe_sidecar_stem()` for derived sidecar file names. [VERIFIED: scripts/run_dataset.py] [VERIFIED: tests/sol_execbench/test_run_dataset_amd_score.py] |
| Claim overstatement in generated reports | Repudiation / Information Integrity | Include explicit claim boundary fields and wording tests. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| Missing evidence presented as passing closure | Information Integrity | Separate `attempted_passed` from `derived_evidence_missing` evidence gap fields. [VERIFIED: .planning/REQUIREMENTS.md] [ASSUMED] |

## Risks

- Existing `traces.json` is a JSON array in `run_dataset.py`, while docs sometimes describe JSONL traces; implementation should read the actual run-dataset output format and avoid renaming canonical artifacts. [VERIFIED: scripts/run_dataset.py] [VERIFIED: docs/analysis.md]
- Ready-subset records include workload UUID and row index, but traces primarily expose `workload.uuid`; row-index fallback must be deterministic for workloads with missing UUIDs. [VERIFIED: src/sol_execbench/core/dataset/ready_subset.py] [VERIFIED: scripts/run_dataset.py]
- Derived sidecars are generated from trace workloads and canonical workload rows; filtered workload paths must be passed to scoring helpers so sidecars match the attempted subset. [VERIFIED: scripts/run_dataset.py]
- `--rerun` deletes prior problem output before execution; closure should not lose skip/failure context if rerun fails after removing old traces. [VERIFIED: scripts/run_dataset.py] [ASSUMED]
- Documentation changes may overlap with Phase 57 claim guardrails, but Phase 55 still needs enough wording to describe bounded closure safely. [VERIFIED: .planning/ROADMAP.md] [VERIFIED: .planning/REQUIREMENTS.md]

## Plan-Shaping Recommendation

Plan the phase in four waves:

1. **Wave 0 tests/schema first:** Add fixture tests for ready-subset loading, filtered workload materialization, no-ready closure, and provenance shape. [ASSUMED]
2. **Wave 1 selection integration:** Add `--ready-subset`, optional `--readiness`, `--execution-closure`, helper functions, and output-only filtered workload files while preserving `run_cli()`. [VERIFIED: scripts/run_dataset.py]
3. **Wave 2 closure joins:** Add trace/summary/log/evidence indexing and per-workload status records for attempted, skipped, failed, missing trace, filtered, and not-attempted states. [VERIFIED: .planning/REQUIREMENTS.md]
4. **Wave 3 guardrails/docs:** Add public contract assertions and `docs/analysis.md` command examples that explicitly reject full validation, paper parity, and leaderboard claims. [VERIFIED: docs/analysis.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

Keep implementation localized to `scripts/run_dataset.py`, tests, and docs unless a tiny dataset closure model under `src/sol_execbench/core/dataset/` becomes necessary for schema reuse. [VERIFIED: scripts/run_dataset.py] [ASSUMED]

## Sources

### Primary (HIGH confidence)
- `.planning/phases/55-ready-subset-selection-and-bounded-execution-closure/55-CONTEXT.md` — locked decisions, scope, status vocabulary, verification constraints.
- `.planning/REQUIREMENTS.md` — EXEC-01 through EXEC-05, claim guardrails, out-of-scope boundaries.
- `.planning/ROADMAP.md` — v1.11 phase ordering and Phase 55 success criteria.
- `src/sol_execbench/core/dataset/ready_subset.py` — ready-subset schema, checksum, claim boundary, workload refs.
- `src/sol_execbench/core/dataset/readiness.py` — readiness statuses, reasons, layered evidence, checksums.
- `scripts/run_dataset.py` — current runner controls, skip/rerun behavior, temporary workload file behavior, trace/summary/evidence outputs.
- `tests/sol_execbench/test_run_dataset_amd_score.py` — monkeypatch runner pattern, skipped-existing-pass derived evidence, sidecar path safety.
- `tests/sol_execbench/test_dataset_inventory_readiness.py` — Phase 54 readiness and ready-subset behavior.
- `tests/sol_execbench/test_public_contract_guardrails.py` — public schema and sidecar-only guardrails.
- `docs/analysis.md` — current dataset, readiness, derived evidence, and claim-boundary documentation.
- `pyproject.toml` — dependency and pytest configuration.

### Secondary (MEDIUM confidence)
- Local environment probes: `uv run python --version`, `uv run pytest --version`, `which uv`.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all recommendations reuse current project dependencies and local environment probes.
- Architecture: HIGH — integration points are directly visible in `run_dataset.py` and Phase 55 context.
- Pitfalls: MEDIUM-HIGH — most are grounded in existing code/tests; some closure vocabulary details are recommended assumptions.

**Research date:** 2026-05-23
**Valid until:** 2026-06-22 for local codebase architecture; revisit if `scripts/run_dataset.py`, Phase 54 sidecars, or public guardrails change.

## RESEARCH COMPLETE
