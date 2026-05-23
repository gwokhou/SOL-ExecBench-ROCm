# Architecture Research: v1.11 Paper Dataset Parity Inventory and ROCm Execution Closure

**Project:** SOL ExecBench ROCm Port  
**Domain:** Dataset parity inventory, ROCm readiness classification, and ready-subset execution reporting  
**Researched:** 2026-05-23  
**Overall confidence:** HIGH for repository integration boundaries and artifact placement; MEDIUM for exact upstream dataset field coverage until live Hugging Face rows are sampled during implementation.

## Executive Summary

v1.11 should add a dataset inventory and parity reporting layer around the existing dataset acquisition and execution path. It should not alter public `Definition`, `Workload`, `Trace`, or `solution.json` schemas, and it should not change the primary `sol-execbench` CLI defaults. The existing architecture already separates canonical evaluation from derived evidence: `scripts/run_dataset.py` shells out to `sol-execbench`, writes per-problem traces and `summary.json`, and optionally emits AMD-native score reports, AMD SOL v2 sidecars, SOLAR derivation sidecars, and timing evidence.

The new work belongs in a dedicated core reporting package, with scripts acting as thin entry points. Inventory generation should inspect dataset layout and canonical problem files, validate them with existing Pydantic schemas where possible, and emit separate JSON reports. ROCm readiness classification should be a pure deterministic transform from file presence, schema parse results, dtype/custom-input/safetensors/reference/solution signals, and optional observed execution failures. Execution closure should then consume the inventory and run only ready or selected problems through the current runner pipeline.

The most important boundary is that inventory and parity reports are sidecars, not benchmark traces. Canonical trace JSONL should remain "what happened when a solution ran against a workload"; parity inventory should answer "what exists, what is missing, what is blocked, and what evidence was produced." This keeps paper dataset parity auditable without turning derived reporting state into public benchmark schema.

## Current Architecture

```text
Acquisition:
  scripts/download_solexecbench.py
    -> Hugging Face nvidia/SOL-ExecBench configs
    -> data/benchmark/<category>/<problem>/
       definition.json
       reference.py
       workload.jsonl

Execution:
  scripts/run_dataset.py
    -> discover_problems()
    -> build reference/custom solution JSON
    -> sol-execbench --definition --workload --solution --json
    -> out/<category>/<problem>/traces.json
    -> out/summary.json

Derived sidecars:
  Definition + Workload + Trace
    -> AMD SOL v2 sidecars
    -> SOLAR derivation sidecars
    -> AMD-native score report
    -> optional rocprofv3 timing evidence
```

Relevant existing contracts:

| Component | Existing responsibility | v1.11 implication |
| --- | --- | --- |
| `src/sol_execbench/core/data/definition.py` | Validates canonical definition fields, tensor dtypes, reference syntax, `run()` parameters, and custom-input entrypoint. | Reuse for schema/readiness validation. Do not add inventory fields here. |
| `src/sol_execbench/core/data/workload.py` | Validates workload axes, random/scalar/safetensors/custom input specs, and custom/non-custom exclusivity. | Reuse to classify custom input and safetensors usage. Do not add readiness fields here. |
| `src/sol_execbench/core/data/trace.py` | Defines canonical evaluation records and status semantics. | Keep stable. Execution closure links to traces by path/ref. |
| `scripts/download_solexecbench.py` | Downloads four public dataset configs and writes the local problem layout. | Add optional manifest/layout verification hooks or move reusable logic into a core dataset module. |
| `scripts/run_dataset.py` | Discovers runnable problems, invokes the primary CLI, and emits traces plus optional derived sidecars. | Add optional inventory filtering and closure report wiring, but keep existing defaults. |
| `src/sol_execbench/core/scoring/amd_score.py` | Builds guarded AMD-native suite reports from traces and sidecars. | Leave as scoring consumer. Inventory should reference score reports, not affect score math. |
| `src/sol_execbench/core/scoring/solar_derivation.py` | Builds strict internal SOLAR derivation sidecars from `Definition` and `Workload`. | Reuse for closure artifacts. Do not merge parity state into SOLAR sidecars. |

## Recommended Architecture

```text
data/benchmark/
  L1/<problem>/
  L2/<problem>/
  Quant/<problem>/
  FlashInfer-Bench/<problem>/

Inventory path:
  dataset layout
    -> DatasetProblemRecord[]
    -> ROCmReadinessClassification[]
    -> PaperParityInventory
    -> ParityGapReport

Execution closure path:
  PaperParityInventory + selected readiness states
    -> ready problem list
    -> existing run_dataset execution loop
    -> canonical traces + summary.json
    -> AMD score / AMD SOL v2 / SOLAR derivation / timing sidecars
    -> ExecutionClosureReport
    -> updated ParityGapReport with observed runtime evidence
```

Recommended module layout:

```text
src/sol_execbench/core/dataset/
  __init__.py
  constants.py            # public categories, expected layout, schema version constants
  discovery.py            # problem directory discovery and category normalization
  inventory.py            # pure file/schema inspection and count aggregation
  readiness.py            # ROCm readiness classification rules
  reports.py              # parity gap and execution closure report dataclasses/parsers
  contracts.py            # strict sidecar JSON parse/serialize helpers
```

Script entry points:

```text
scripts/download_solexecbench.py
  -> keep dataset download role
  -> optionally write acquisition manifest

scripts/run_dataset.py
  -> keep execution role
  -> optionally consume inventory / emit closure report

scripts/inventory_solexecbench.py
  -> new thin CLI for layout verification, inventory JSON, gap report JSON
```

Keep the core implementation importable and testable without Hugging Face, ROCm, or subprocess execution. The scripts should mostly parse CLI arguments, call pure helpers, and write JSON.

## New Modules

| Module | Responsibility | Notes |
| --- | --- | --- |
| `src/sol_execbench/core/dataset/constants.py` | Define `PAPER_DATASET_CATEGORIES = ("L1", "L2", "Quant", "FlashInfer-Bench")` and sidecar schema versions. | Replace duplicate category constants in scripts over time. |
| `src/sol_execbench/core/dataset/discovery.py` | Discover single problem dirs or dataset roots, preserving current `run_dataset.py` behavior. | Move or wrap `discover_problems()` here, then import from scripts. |
| `src/sol_execbench/core/dataset/inventory.py` | Build per-problem inventory records from local files and schema parsing. | No subprocess or GPU dependency. |
| `src/sol_execbench/core/dataset/readiness.py` | Classify each problem as ready or blocked with machine-readable reasons. | Pure function over inventory records and optional observed results. |
| `src/sol_execbench/core/dataset/reports.py` | Build suite-level inventory, gap, and execution closure reports. | Dataclasses with `to_dict()` and strict parsers, matching scoring sidecar style. |
| `scripts/inventory_solexecbench.py` | User-facing inventory/report command. | Additive script; does not affect `sol-execbench`. |

## Modified Modules

| Module | Modification | Boundary |
| --- | --- | --- |
| `scripts/download_solexecbench.py` | Add `--output`, `--category`, `--manifest`, and `--verify-layout` only if needed; reuse category constants. | Still only acquisition/layout. No evaluation. |
| `scripts/run_dataset.py` | Import shared discovery; add optional `--inventory`, `--readiness ready`, `--closure-report`, and `--parity-gap-report` flags. | Existing invocation and output remain valid. |
| `src/sol_execbench/core/scoring/amd_score.py` | No structural change expected. At most ensure report refs can be included from closure report. | Score layer must not inspect inventory state. |
| `src/sol_execbench/core/scoring/solar_derivation.py` | No change expected. Continue generating sidecars from canonical inputs. | Keep sidecar contract independent of parity reports. |
| Tests | Add dataset inventory/readiness/report tests and run_dataset filter tests. | Prefer CPU-only fixtures. |
| Docs | Add internal dataset parity report documentation and claim guardrails. | Must distinguish inventory completion from full 235-problem ROCm validation. |

## Artifact Locations

Recommended default outputs:

```text
out/dataset-parity/
  acquisition-manifest.json
  paper-parity-inventory.json
  rocm-readiness.json
  parity-gap-report.json
  execution-closure-report.json

out/<category>/<problem>/
  traces.json
  solution.json
  *_cli.log

out/amd-sol-v2/
  <definition>.<workload>.amd-sol-v2.json

out/solar-derivation/
  <definition>.<workload>.solar-derivation.json

out/timing-evidence/
  <category>/<problem>.timing.json

out/amd-score-report.json
```

Do not write inventory fields into:

- `definition.json`
- `workload.jsonl`
- canonical trace JSONL / `traces.json`
- `solution.json`
- AMD SOL v2 sidecars
- SOLAR derivation sidecars

Those artifacts should instead be referenced from closure/gap reports using stable relative paths.

## JSON Contracts

### Acquisition Manifest

Schema version: `sol_execbench.dataset_acquisition.v1`

Purpose: record what was downloaded or layout-verified.

Required top-level fields:

```json
{
  "schema_version": "sol_execbench.dataset_acquisition.v1",
  "dataset": "nvidia/SOL-ExecBench",
  "root": "data/benchmark",
  "categories": ["L1", "L2", "Quant", "FlashInfer-Bench"],
  "generated_at": "ISO-8601",
  "source": {
    "kind": "huggingface|local_layout",
    "repo_id": "nvidia/SOL-ExecBench",
    "configs": ["L1", "L2", "Quant", "FlashInfer-Bench"]
  },
  "category_counts": {"L1": 0, "L2": 0, "Quant": 0, "FlashInfer-Bench": 0},
  "warnings": []
}
```

### Paper Parity Inventory

Schema version: `sol_execbench.paper_parity_inventory.v1`

Purpose: one machine-readable record per local public benchmark problem.

Per-problem record:

```json
{
  "category": "L1",
  "problem": "name",
  "path": "data/benchmark/L1/name",
  "definition_name": "name",
  "files": {
    "definition_json": true,
    "workload_jsonl": true,
    "reference_py": true,
    "solution_json": false,
    "solution_py": false
  },
  "schema": {
    "definition_valid": true,
    "workloads_valid": true,
    "workload_count": 1,
    "errors": []
  },
  "coverage": {
    "dtypes": ["float16"],
    "input_kinds": ["random"],
    "uses_custom_inputs": false,
    "uses_safetensors": false,
    "has_forward_indicator": true,
    "has_backward_indicator": false,
    "op_type": "matmul"
  },
  "availability": {
    "reference_available": true,
    "solution_available": false
  }
}
```

Suite summary:

```json
{
  "schema_version": "sol_execbench.paper_parity_inventory.v1",
  "generated_at": "ISO-8601",
  "dataset_root": "data/benchmark",
  "expected_categories": ["L1", "L2", "Quant", "FlashInfer-Bench"],
  "category_counts": {},
  "total_problems": 0,
  "records": []
}
```

Forward/backward indicators should be conservative. Prefer explicit dataset fields if acquisition preserves them; otherwise derive from `definition.name`, `op_type`, description, and reference signatures with a `derived_indicator` warning. Do not overstate this as an upstream label unless it is actually present in the downloaded row.

### ROCm Readiness Report

Schema version: `sol_execbench.rocm_readiness.v1`

Allowed readiness states:

- `ready`
- `schema_input_blocked`
- `dtype_blocked`
- `custom_input_blocked`
- `runtime_blocked`
- `unsupported_nvidia_only_path`
- `needs_hardware_evidence`

Per-problem record:

```json
{
  "category": "L1",
  "problem": "name",
  "readiness": "ready",
  "reasons": [],
  "recommended_action": "run_small_batch",
  "evidence": {
    "inventory_ref": "paper-parity-inventory.json#records/L1/name",
    "definition_ref": "data/benchmark/L1/name/definition.json",
    "workload_ref": "data/benchmark/L1/name/workload.jsonl"
  }
}
```

Suggested classification rules:

| State | Rule |
| --- | --- |
| `schema_input_blocked` | Missing files, invalid `Definition`, invalid `Workload`, malformed JSON, mixed custom/non-custom inputs, unresolved workload axes. |
| `dtype_blocked` | Definition or workload requires dtype unsupported by the ROCm port or current PyTorch ROCm path, especially paper-only low precision formats without implemented AMD support. |
| `custom_input_blocked` | Custom input entrypoint is present but cannot be validated or safely executed in inventory-only mode. |
| `unsupported_nvidia_only_path` | Reference or solution source imports CUDA/NVIDIA-only libraries or uses CUDA-only APIs with no ROCm replacement. |
| `runtime_blocked` | A previous closure run produced compile/runtime/timeout failures. |
| `needs_hardware_evidence` | Schema appears runnable, but no closure trace exists for the target hardware class. |
| `ready` | Required files parse, reference exists, workloads parse, input kinds are supported, and no static NVIDIA-only blocker is detected. |

`ready` should mean "eligible for small closure execution," not "validated across all workloads and hardware."

### Execution Closure Report

Schema version: `sol_execbench.execution_closure.v1`

Purpose: connect readiness to observed run artifacts.

Required fields:

```json
{
  "schema_version": "sol_execbench.execution_closure.v1",
  "generated_at": "ISO-8601",
  "dataset_root": "data/benchmark",
  "output_root": "out",
  "selection": {
    "categories": ["L1"],
    "readiness": ["ready"],
    "limit": 5,
    "max_workloads": 1
  },
  "summary": {
    "selected_problems": 5,
    "executed_problems": 5,
    "passed_problems": 4,
    "failed_problems": 1,
    "trace_count": 5
  },
  "records": [
    {
      "category": "L1",
      "problem": "name",
      "readiness_before_run": "ready",
      "execution_status": "passed|failed|skipped",
      "trace_ref": "L1/name/traces.json",
      "summary_ref": "summary.json#L1/name",
      "amd_score_ref": "amd-score-report.json#scores/name",
      "amd_sol_refs": [],
      "solar_derivation_refs": [],
      "timing_evidence_ref": null,
      "warnings": []
    }
  ],
  "claim_level": "inventory-and-small-batch-rocm-closure",
  "non_claims": [
    "not full 235-problem ROCm validation",
    "not upstream SOLAR equivalence",
    "not hosted leaderboard equivalence",
    "not CDNA3 hardware validation unless separately evidenced"
  ]
}
```

### Parity Gap Report

Schema version: `sol_execbench.parity_gap_report.v1`

Purpose: aggregate missing inventory, blockers, and closure gaps.

Required sections:

```json
{
  "schema_version": "sol_execbench.parity_gap_report.v1",
  "generated_at": "ISO-8601",
  "inventory_ref": "paper-parity-inventory.json",
  "readiness_ref": "rocm-readiness.json",
  "closure_ref": "execution-closure-report.json",
  "counts": {
    "inventory_total": 0,
    "ready": 0,
    "blocked": 0,
    "executed": 0,
    "passed": 0,
    "failed": 0
  },
  "gaps_by_category": {},
  "gaps_by_reason": {},
  "claim_guardrails": [],
  "next_actions": []
}
```

## Data Flow Details

### Inventory Build

1. Discover categories from the expected public set: `L1`, `L2`, `Quant`, `FlashInfer-Bench`.
2. For every problem directory, check file presence for `definition.json`, `workload.jsonl`, `reference.py`, `solution.json`, and `solution.py`.
3. Parse `definition.json` with `Definition`.
4. Parse every workload JSONL line with `Workload`.
5. Compute inventory features:
   - dtype set from `Definition.inputs` and `Definition.outputs`
   - input kinds from workload input specs
   - custom-input usage from `Definition.custom_inputs_entrypoint` and `CustomInput`
   - safetensors usage from `SafetensorsInput`
   - reference availability from `Definition.reference` and/or `reference.py`
   - solution availability from local solution files
   - op/category hints from `op_type`, name, description, and optional upstream fields if preserved
6. Emit inventory JSON.

### Readiness Classification

1. Start from inventory record.
2. If canonical files are missing or schema parsing failed, classify `schema_input_blocked`.
3. If dtype set includes known unsupported low precision or packed formats for the current ROCm path, classify `dtype_blocked`.
4. If custom input is present and there is no validated custom-input execution path for that pattern, classify `custom_input_blocked`.
5. If static source scan finds CUDA/NVIDIA-only APIs in reference or solution files, classify `unsupported_nvidia_only_path`.
6. If prior execution evidence exists with runtime failure, classify `runtime_blocked`.
7. If runnable but no target-hardware trace evidence exists, classify `needs_hardware_evidence`.
8. Otherwise classify `ready`.

The classifier should support both strict mode and closure mode. In strict inventory mode, "needs hardware evidence" is acceptable for otherwise runnable problems. In execution selection mode, callers may choose to run both `ready` and `needs_hardware_evidence` records, but the report must preserve the distinction.

### Execution Closure

1. Load inventory/readiness report or build one in memory.
2. Select records by category, readiness state, limit, and optional max workloads.
3. Reuse `scripts/run_dataset.py` execution loop:
   - build reference or custom solution
   - call `sol-execbench --json`
   - save `traces.json`
   - inspect traces
   - optionally emit AMD score, AMD SOL v2, SOLAR derivation, and timing evidence
4. Write `execution-closure-report.json`.
5. Update or generate `parity-gap-report.json`.

Do not bypass `sol-execbench` for closure runs. The purpose of closure is to prove the existing public execution path can run selected public dataset problems.

## Build Order

1. **Dataset Contract and Discovery**
   - Add `core/dataset/constants.py` and `core/dataset/discovery.py`.
   - Move duplicate category/discovery behavior out of `scripts/run_dataset.py` behind compatible imports.
   - Add tests that current dataset discovery behavior remains unchanged.

2. **Inventory Records**
   - Add `core/dataset/inventory.py` with strict dataclasses and `to_dict()`.
   - Parse local fixture problem dirs through `Definition` and `Workload`.
   - Emit `paper-parity-inventory.json`.

3. **Readiness Classification**
   - Add `core/dataset/readiness.py`.
   - Implement deterministic blocker rules with reason codes.
   - Add fixtures for missing files, invalid schema, safetensors, custom input, unsupported dtype, CUDA-only source, and ready problem.

4. **Inventory CLI**
   - Add `scripts/inventory_solexecbench.py`.
   - Support `--dataset-root`, `--output-dir`, `--category`, `--json`, and `--gap-report`.
   - Keep this separate from `sol-execbench`.

5. **Runner Selection and Closure Report**
   - Extend `scripts/run_dataset.py` with optional inventory/readiness filtering and `--closure-report`.
   - Reuse existing sidecar options without changing defaults.
   - Ensure skipped existing traces can still produce closure records and derived reports.

6. **Gap Report and Claim Guardrails**
   - Add `core/dataset/reports.py` aggregation.
   - Add docs/tests that distinguish inventory completion, small-batch execution closure, full 235-problem validation, CDNA3 validation, upstream SOLAR equivalence, and leaderboard equivalence.

7. **Acquisition Manifest**
   - Extend `scripts/download_solexecbench.py` last, after inventory expectations are stable.
   - Add manifest output for downloaded/local-layout verified datasets.

## Patterns to Follow

### Sidecar-Only Reporting

**What:** Dataset parity, readiness, closure, and gap reports are separate JSON artifacts with their own schema versions.

**Why:** The public benchmark contracts are already stable and narrow. Adding parity fields to traces or problem schemas would create compatibility churn and blur observed execution with derived reporting.

### Pure Core, Thin Scripts

**What:** Put discovery, inventory, classification, and report assembly in `src/sol_execbench/core/dataset/`; keep scripts as argument parsing and file writing.

**Why:** CPU-only tests can cover most v1.11 behavior without Hugging Face, ROCm hardware, or subprocess execution.

### Evidence References Instead of Embedding

**What:** Closure and gap reports should link to traces, score reports, SOL sidecars, SOLAR derivation sidecars, and timing evidence by relative path and optional fragment.

**Why:** Avoids duplicating large artifacts and keeps each sidecar contract independently parseable.

### Conservative Classification

**What:** When evidence is incomplete, classify as blocked or needing hardware evidence rather than ready.

**Why:** The milestone goal is auditable parity, not maximizing the ready count.

## Anti-Patterns to Avoid

### Mutating Canonical Trace JSONL

**What goes wrong:** Adding inventory/readiness/parity fields to `Trace` creates a public schema change.

**Instead:** Put all parity metadata in sidecar reports that reference trace paths.

### Expanding the Primary CLI

**What goes wrong:** Adding dataset parity behavior to `sol-execbench` makes a single-problem evaluator responsible for dataset research workflows.

**Instead:** Use `scripts/inventory_solexecbench.py` and optional `scripts/run_dataset.py` flags.

### Treating Inventory as Validation

**What goes wrong:** A complete inventory can be misread as a full ROCm benchmark pass.

**Instead:** Reports must carry explicit claim levels and non-claims.

### Running Blocked Problems Blindly

**What goes wrong:** Batch runs spend time on known schema/dtype/custom-input blockers and produce noisy runtime failures.

**Instead:** Run selected readiness states intentionally and preserve skipped/blocker evidence in closure reports.

## Scalability Considerations

| Concern | Small fixture set | Public dataset scale | Future full validation |
| --- | --- | --- | --- |
| Inventory parsing | In-memory list is fine. | In-memory list of hundreds of records is fine. | Keep JSONL option optional if records grow beyond paper dataset scale. |
| Execution | Existing `--limit` and `--max-workloads`. | Add readiness filter to avoid known blockers. | Add resume/retry and per-category shards if full validation is attempted. |
| Sidecars | Simple output dirs. | Use safe sidecar filenames already present in runner. | Add manifest index if sidecar count becomes hard to inspect. |
| Hardware evidence | Usually absent. | Record `needs_hardware_evidence`. | Split reports by architecture, e.g. `gfx1200`, `gfx94*`. |
| Claim control | Static docs/tests. | Report-level non-claims. | Require hardware-specific evidence manifests before validation claims. |

## Testing Strategy

Recommended focused tests:

| Test file | Coverage |
| --- | --- |
| `tests/sol_execbench/test_dataset_discovery.py` | Shared discovery preserves current single-root and category behavior. |
| `tests/sol_execbench/test_paper_parity_inventory.py` | Inventory fields, counts, dtype/input-kind/custom/safetensors/reference/solution detection. |
| `tests/sol_execbench/test_rocm_readiness.py` | Every readiness state and reason code. |
| `tests/sol_execbench/test_dataset_reports.py` | Strict schema versions, report aggregation, claim guardrails, artifact refs. |
| `tests/sol_execbench/test_run_dataset_closure.py` | Runner filters by readiness and writes closure report using mocked CLI results. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | No new fields in canonical trace/definition/workload contracts; primary CLI remains stable. |

Most tests should use temporary fixture problem directories. Avoid network and GPU dependencies except for explicit integration checks.

## Source-Grounded Confidence

| Finding | Confidence | Basis |
| --- | --- | --- |
| Inventory should be sidecar-only | HIGH | Existing `Definition`, `Workload`, and `Trace` schemas are public execution contracts; project explicitly requires stability. |
| Runner is the right execution closure integration point | HIGH | `scripts/run_dataset.py` already owns discovery, CLI subprocess execution, summary, traces, AMD score, SOL v2, SOLAR derivation, and timing evidence. |
| New core dataset package is preferable to adding more script logic | HIGH | Existing runner is already large; pure helpers improve testability and keep scripts thin. |
| Exact upstream row metadata for forward/backward indicators may need validation | MEDIUM | `download_solexecbench.py` currently preserves only selected row fields and does not retain all possible upstream metadata. |
| Readiness states listed in the milestone are sufficient | MEDIUM-HIGH | States map cleanly to existing schema/input/runtime boundaries, but implementation may discover finer reason codes. |

## Open Questions For Implementation

- Does the Hugging Face dataset expose explicit forward/backward indicators, or must they be derived from names/descriptions/reference code?
- Should `download_solexecbench.py` preserve the original raw row metadata in an optional sidecar for auditability?
- Which dtype values are considered "blocked" versus "needs hardware evidence" for ROCm >= 7.0 on RDNA 4 and CDNA 3?
- Should custom-input problems be blocked by default until a fixture proves the entrypoint is safe and deterministic?
- Should closure reports be architecture-specific from the start, or is hardware architecture captured only through trace environments until full validation work begins?

## Recommended Roadmap Implication

Build v1.11 in this order: shared discovery, inventory, readiness classification, inventory CLI, runner closure integration, gap reports, acquisition manifest. This order keeps all early work CPU-only and schema-focused, then plugs into the existing execution and scoring sidecar pipeline only after selection and report contracts are stable.
