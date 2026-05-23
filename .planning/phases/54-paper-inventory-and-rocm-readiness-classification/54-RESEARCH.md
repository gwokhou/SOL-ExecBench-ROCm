# Phase 54: Paper Inventory And ROCm Readiness Classification - Research

**Researched:** 2026-05-23
**Domain:** Dataset inventory, static ROCm readiness classification, sidecar artifact generation
**Confidence:** HIGH

## User Constraints (from CONTEXT.md)

### Locked Decisions

- Use the Phase 53 dataset root and manifest as entry context, but re-read
  canonical `definition.json` and `workload.jsonl` files as the source of truth.
- Schema parsing failures should produce structured `schema_failure` records
  and denominator counts without aborting the whole inventory run.
- Record reference and solution availability by path only; do not import,
  execute, or inspect runtime behavior.
- Model inventory as problem-level records with workload-level child records,
  including UUIDs, input kinds, dtype/shape hints, custom input usage, and
  safetensors usage at workload granularity.
- Classify readiness primarily at workload level, with problem-level summaries
  derived from the most blocking workload states.
- Define `ready` as "ready to attempt local ROCm execution", not as validated
  or passed. It requires parseable schema, locatable/generatable inputs,
  reference availability, and no known NVIDIA-only/runtime blocker.
- Represent low-precision and Quant readiness in layers: schema-known,
  input-generation, reference-execution, candidate-execution, and
  hardware-validation evidence.
- Treat custom inputs and safetensors as explicit asset/evaluator requirements;
  missing local assets or unsupported evaluator paths become blockers rather
  than random substitution.
- Extend `src/sol_execbench/core/dataset/` with inventory and readiness modules;
  keep scripts as thin CLI wrappers.
- Emit deterministic `inventory.json`, `readiness.json`, and
  `ready_subset.json` sidecar artifacts with schema versions.
- Add a thin `scripts/inspect_dataset.py` CLI supporting `--dataset-root`,
  `--manifest`, `--inventory`, `--readiness`, and `--ready-subset`.
- Re-export stable helper/model names from `sol_execbench.core.dataset` only;
  do not change the primary `sol-execbench` CLI or public benchmark schemas.
- Use fixture tests covering schema-ok ready, schema failure, missing files,
  custom inputs, safetensors, unsupported dtype/Quant, NVIDIA-only hints, and
  hardware-evidence-needed cases.
- Do not run real GPU/ROCm evaluation in Phase 54.
- Generate ready-subset manifests from readiness results only, including stable
  references for `ready` workloads and no canonical dataset mutations.
- Add tests and docs wording that readiness means "can attempt execution", not
  passed, scored, fully validated, or paper-parity complete.

### the agent's Discretion

The agent may choose exact module names, enum/model shapes, status severity
ordering, and CLI formatting as long as outputs are deterministic, sidecar-only,
and preserve the locked scope boundaries.

### Deferred Ideas (OUT OF SCOPE)

- Ready-subset execution is deferred to Phase 55.
- Parity gap report aggregation is deferred to Phase 56.
- Milestone release claim closure is deferred to Phase 57.
- Full public dataset real-hardware validation remains out of scope.

## Summary

Phase 54 should add pure, deterministic dataset-sidecar modules under `src/sol_execbench/core/dataset/`: `inventory.py` for canonical schema parsing and denominator accounting, `readiness.py` for workload-first static classification, and `ready_subset.py` for sidecar generation. This fits the Phase 53 package boundary, where `layout.py` performs shallow file discovery and `manifest.py` serializes deterministic sidecar JSON with schema versions and checksums. [VERIFIED: src/sol_execbench/core/dataset/layout.py] [VERIFIED: src/sol_execbench/core/dataset/manifest.py]

The implementation must parse `definition.json` through `Definition` and each non-empty workload JSONL row through `Workload`; those are the authoritative public contracts and already validate reference syntax, `run()` parameter order, axis references, custom input entrypoints, workload input variants, and no mixed custom/non-custom inputs. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py]

**Primary recommendation:** Implement a three-artifact pipeline: `build_dataset_inventory(root, manifest=None)` -> `classify_rocm_readiness(inventory)` -> `build_ready_subset(readiness)`, exposed by a thin `scripts/inspect_dataset.py` wrapper and covered by fixture-only pytest tests. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md]

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INV-01 | Generate machine-readable inventory for every discovered problem/workload using current Pydantic contracts. | Use `Definition(**payload)` and `Workload(**row)` as the only schema parsers; persist failures as records. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py] |
| INV-02 | Record category, problem path, workload UUIDs/count, dtypes, input kinds, custom inputs, safetensors, reference, solution availability. | Existing models expose `inputs`, `outputs`, `custom_inputs_entrypoint`, workload `uuid`, and input variant `type`; `reference.py`/solution files can be path-checked only. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py] [VERIFIED: scripts/run_dataset.py] |
| INV-03 | Record forward/backward and operation-family hints when present or conservatively derived. | `Definition.op_type`, `Definition.name`, `description`, and category/path are available; unknowns should remain explicit. [VERIFIED: src/sol_execbench/core/data/definition.py] |
| INV-04 | Expose category/suite denominators for discovered, parsed, schema failures, and missing required files. | Phase 53 layout already counts present problems/workloads and missing categories; Phase 54 should add parse-level denominators. [VERIFIED: src/sol_execbench/core/dataset/layout.py] |
| INV-05 | Deterministic inventory shape and identifiers. | Existing dataset helpers sort categories/problems and serialize JSON with `sort_keys=True`; reuse that convention. [VERIFIED: src/sol_execbench/core/dataset/categories.py] [VERIFIED: src/sol_execbench/core/dataset/manifest.py] |
| READY-01 | Classify deterministic readiness statuses. | Implement a fixed severity order and status enum matching the requirement vocabulary. [VERIFIED: .planning/REQUIREMENTS.md] |
| READY-02 | Record reason codes, evidence paths, next actions. | Existing sidecars use explicit references and no canonical schema mutation; readiness should follow the same derived-artifact pattern. [VERIFIED: docs/rocm_timing.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| READY-03 | Treat custom inputs/safetensors as explicit requirements. | `CustomInput` and `SafetensorsInput` are first-class workload variants; `gen_inputs()` raises when required custom/safetensors inputs are not supplied. [VERIFIED: src/sol_execbench/core/data/workload.py] [VERIFIED: src/sol_execbench/core/bench/io.py] |
| READY-04 | Distinguish schema-known, input-generation, reference-execution, candidate-execution, hardware-validation evidence for low precision/Quant. | DType includes FP8/FP4, `dtype_str_to_torch_dtype()` maps them, and `_rand_tensor()` has special paths; readiness still needs layered evidence because static dtype availability is not execution validation. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/dtypes.py] [VERIFIED: src/sol_execbench/core/bench/io.py] |
| READY-05 | Generate ready-subset manifest without canonical mutations. | Phase 53 sidecars are separate JSON artifacts; public guardrails protect canonical schemas and CLI. [VERIFIED: src/sol_execbench/core/dataset/manifest.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |

## Project Constraints (from AGENTS.md)

- Source code belongs under `src/sol_execbench/`; tests belong under `tests/sol_execbench/`; scripts belong under `scripts/`. [VERIFIED: AGENTS.md]
- Use Python 3.12+ and Ruff style; keep changes consistent with nearby modules and avoid broad refactors. [VERIFIED: AGENTS.md] [VERIFIED: pyproject.toml]
- Use pytest; add small unit tests for schema/driver logic and integration coverage only when subprocess/GPU behavior changes. [VERIFIED: AGENTS.md]
- Environment-sensitive GPU tests must use existing markers such as `requires_rocm`, `requires_rdna4`, and `requires_cdna3`; Phase 54 should not need them. [VERIFIED: AGENTS.md] [VERIFIED: tests/conftest.py]
- Do not commit credentials, proprietary kernels, Hugging Face tokens, downloaded datasets, cache, build, or benchmark output. [VERIFIED: AGENTS.md]
- Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable. [VERIFIED: AGENTS.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- Before repo edits, use a GSD workflow; this research artifact is part of the GSD plan-phase workflow. [VERIFIED: AGENTS.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Dataset discovery | Filesystem / Data Layer | CLI wrapper | Category/problem discovery is local path traversal over dataset roots. [VERIFIED: src/sol_execbench/core/dataset/layout.py] |
| Canonical schema parsing | Core Data Layer | — | `Definition` and `Workload` own public dataset validation. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py] |
| Inventory sidecar generation | Dataset Core | Filesystem output | This is a pure transformation from canonical files to derived JSON. [VERIFIED: src/sol_execbench/core/dataset/manifest.py] |
| ROCm readiness classification | Dataset Core | Bench IO metadata | Static classification should inspect metadata and known input-generation boundaries without executing GPU code. [VERIFIED: src/sol_execbench/core/bench/io.py] |
| Ready subset generation | Dataset Core | Phase 55 runner | Phase 54 selects ready references; Phase 55 executes through `scripts/run_dataset.py`. [VERIFIED: .planning/ROADMAP.md] [VERIFIED: scripts/run_dataset.py] |
| Human CLI | Script Layer | Dataset Core | `scripts/download_solexecbench.py` already uses a thin wrapper around dataset helpers; mirror that style. [VERIFIED: scripts/download_solexecbench.py] |

## Existing Schema Details

### Definition

`Definition` fields are `name`, `op_type`, `axes`, `custom_inputs_entrypoint`, `inputs`, `outputs`, `reference`, `description`, and `hf_id`; public guardrails assert that exact top-level key set. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

Axis variants are `const`, `var`, and `expr`; tensor specs include `shape` and `dtype`; scalar tensors are represented by `shape=None`. [VERIFIED: src/sol_execbench/core/data/definition.py]

`DType` currently includes float64/32/16, bfloat16, FP8 E4M3/E5M2, FP4 E2M1 variants, int64/32/16/8, and bool. [VERIFIED: src/sol_execbench/core/data/definition.py]

Definition validation parses the embedded `reference` Python code, requires a top-level `run` function, requires `run()` parameters to match input keys in order, validates custom input entrypoint existence, forbids input names that collide with axes, forbids input/output name overlap, and validates tensor shape axis references. [VERIFIED: src/sol_execbench/core/data/definition.py]

### Workload

`Workload` fields are `axes`, `inputs`, `uuid`, and `tolerance`; public guardrails assert that exact top-level key set. [VERIFIED: src/sol_execbench/core/data/workload.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

Input variants are `random`, `scalar`, `safetensors`, and `custom`; `SafetensorsInput` carries `path` and `tensor_key`; `CustomInput` requires definition-level custom generation metadata by convention and validator message. [VERIFIED: src/sol_execbench/core/data/workload.py]

Workload validation rejects mixing any `custom` input with non-custom inputs. [VERIFIED: src/sol_execbench/core/data/workload.py]

### Bench IO Implications

`load_safetensors()` resolves safetensors paths against blob roots, validates tensor key, shape, and dtype, and raises on missing files/keys or mismatch. [VERIFIED: src/sol_execbench/core/bench/io.py]

`gen_inputs()` can generate random/scalar inputs, uses custom input factories when supplied, and raises when required safetensors or custom tensors are missing. [VERIFIED: src/sol_execbench/core/bench/io.py]

Low-precision random generation has explicit FP8 and FP4 paths, but that only proves an input-generation path exists for supported torch dtypes; it does not prove reference execution, candidate execution, numerical correctness, hardware validation, or paper parity. [VERIFIED: src/sol_execbench/core/bench/io.py] [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md]

## Inventory Model Proposal

Use Pydantic `BaseModel` models in `src/sol_execbench/core/dataset/inventory.py`, because Phase 53 dataset sidecars already use Pydantic models for deterministic JSON payloads. [VERIFIED: src/sol_execbench/core/dataset/manifest.py]

Recommended schema constants:

- `INVENTORY_SCHEMA_VERSION = "sol_execbench.dataset_inventory.v1"` [ASSUMED]
- `READINESS_SCHEMA_VERSION = "sol_execbench.rocm_readiness.v1"` [ASSUMED]
- `READY_SUBSET_SCHEMA_VERSION = "sol_execbench.ready_subset.v1"` [ASSUMED]

Recommended top-level inventory shape:

```json
{
  "schema_version": "sol_execbench.dataset_inventory.v1",
  "created_at": "2026-05-23T00:00:00Z",
  "root": {"path": "..."},
  "manifest_ref": {"path": "...", "checksum": "..."},
  "selected_categories": ["FlashInfer-Bench", "L1", "L2", "Quant"],
  "categories": [],
  "problems": [],
  "denominators": {},
  "diagnostics": [],
  "inventory_checksum": {"algorithm": "sha256", "value": "..."}
}
```

The `created_at` field should be injectable in tests like `build_dataset_manifest(..., created_at=...)`; checksum calculation should null the checksum field and use `stable_json_checksum()`. [VERIFIED: src/sol_execbench/core/dataset/manifest.py] [ASSUMED]

Recommended problem record:

```json
{
  "category": "L1",
  "problem_id": "L1/matmul_demo",
  "problem_path": "L1/matmul_demo",
  "definition_path": "L1/matmul_demo/definition.json",
  "workload_path": "L1/matmul_demo/workload.jsonl",
  "reference_path": "L1/matmul_demo/reference.py",
  "reference_available": true,
  "solution_files": ["solution.json", "solution.py"],
  "schema_status": "parsed",
  "schema_failure": null,
  "definition": {
    "name": "matmul_demo",
    "op_type": "matmul",
    "op_family_hint": "matmul",
    "op_family_source": "definition.op_type",
    "direction_hint": "unknown",
    "direction_source": "unknown",
    "input_dtypes": ["float32"],
    "output_dtypes": ["float32"],
    "input_shapes": {"x": ["M", "N"]},
    "output_shapes": {"out": ["M", "N"]},
    "custom_inputs_entrypoint": null
  },
  "workload_count": 1,
  "workloads": []
}
```

Recommended workload child record:

```json
{
  "uuid": "workload-1",
  "row_index": 1,
  "axes": {"M": 16},
  "input_kinds": {"x": "random"},
  "input_kind_counts": {"random": 1, "scalar": 0, "safetensors": 0, "custom": 0},
  "uses_custom_inputs": false,
  "uses_safetensors": false,
  "safetensors_refs": [],
  "scalar_inputs": {},
  "input_dtypes": {"x": "float32"},
  "output_dtypes": {"out": "float32"},
  "resolved_input_shapes": {"x": [16]},
  "resolved_output_shapes": {"out": [16]},
  "shape_status": "resolved"
}
```

If definition parsing fails, emit a problem-level `schema_status="schema_failure"`, record the Pydantic error string/code as diagnostic detail, and skip workload parsing for readiness while still counting discovered problem and missing/parse failure denominators. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: src/sol_execbench/core/data/definition.py]

If a workload row fails parsing, keep the problem parsed but emit workload child status `schema_failure` with `row_index`, `line_ref`, and error details. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] [ASSUMED]

## Readiness Status And Reason-Code Proposal

Readiness should be workload-first, with a derived problem status computed from the most blocking workload state. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md]

Use this fixed severity order:

1. `schema_input_blocked`
2. `unsupported_nvidia_only_path`
3. `custom_input_blocked`
4. `dtype_blocked`
5. `runtime_blocked`
6. `needs_hardware_evidence`
7. `ready`

This order is conservative: parse and asset blockers prevent meaningful attempts; hardware evidence is not a blocker to "attempt local ROCm execution" unless the workload uses known low-precision/Quant paths that require explicit validation. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] [ASSUMED]

Recommended stable reason codes:

| Status | Reason Code | Trigger | Next Action |
|--------|-------------|---------|-------------|
| `schema_input_blocked` | `definition_schema_failure` | `Definition` parse failed | Fix or exclude malformed canonical problem. [VERIFIED: src/sol_execbench/core/data/definition.py] |
| `schema_input_blocked` | `workload_schema_failure` | `Workload` parse failed | Fix or exclude malformed workload row. [VERIFIED: src/sol_execbench/core/data/workload.py] |
| `schema_input_blocked` | `missing_definition_json` | Required definition file absent | Acquire or restore dataset file. [VERIFIED: src/sol_execbench/core/dataset/layout.py] |
| `schema_input_blocked` | `missing_workload_jsonl` | Required workload file absent | Acquire or restore dataset file. [VERIFIED: src/sol_execbench/core/dataset/layout.py] |
| `schema_input_blocked` | `missing_reference` | No embedded definition reference or reference path unavailable | Restore reference source before execution attempt. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: scripts/run_dataset.py] |
| `unsupported_nvidia_only_path` | `nvidia_cuda_runtime_hint` | Static reference text contains NVIDIA-only runtime/tooling hints outside allowed PyTorch ROCm compatibility names | Port reference/evaluator path or classify out of ready subset. [VERIFIED: docs/rocm_timing.md] [ASSUMED] |
| `custom_input_blocked` | `custom_input_requires_entrypoint` | Workload uses custom input but entrypoint missing/unusable after schema parsing | Add valid custom input entrypoint or block. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py] |
| `custom_input_blocked` | `custom_input_requires_evaluator_support` | Custom input generation would require executing definition code in Phase 54 | Defer to execution phase; do not random-substitute. [VERIFIED: src/sol_execbench/core/bench/io.py] [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] |
| `runtime_blocked` | `safetensors_asset_missing` | Safetensors path cannot be found under dataset/blob roots | Acquire asset or configure blob root. [VERIFIED: src/sol_execbench/core/bench/io.py] |
| `runtime_blocked` | `safetensors_key_unknown_static` | Static inventory cannot prove tensor key availability without reading blob | Defer to execution or optional metadata probe. [ASSUMED] |
| `dtype_blocked` | `unsupported_dtype_for_input_generation` | DType cannot map to current torch dtype/input generation path | Add generator support or classify unsupported. [VERIFIED: src/sol_execbench/core/data/dtypes.py] [VERIFIED: src/sol_execbench/core/bench/io.py] |
| `needs_hardware_evidence` | `low_precision_requires_hardware_evidence` | Any FP8/FP4 dtype or `Quant` category workload is otherwise ready | Attempt only with explicit hardware evidence in later phase. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: .planning/REQUIREMENTS.md] |
| `ready` | `ready_to_attempt_rocm_execution` | Parseable, inputs locatable/generatable, reference available, no static blocker | Eligible for ready subset. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] |

Represent layered readiness as booleans or enum states on each workload:

```json
{
  "schema_known": "ok",
  "input_generation": "ok|blocked|needs_asset",
  "reference_execution": "ready_to_attempt|blocked|unknown",
  "candidate_execution": "not_evaluated",
  "hardware_validation": "not_required|needed|not_evaluated"
}
```

`candidate_execution` must remain `not_evaluated` in Phase 54 because this phase does not inspect or run candidate solutions. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md]

## Ready-Subset Shape

Generate `ready_subset.json` strictly from `readiness.json`; never rescan the dataset for hidden eligibility decisions. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md]

Recommended shape:

```json
{
  "schema_version": "sol_execbench.ready_subset.v1",
  "created_at": "2026-05-23T00:00:00Z",
  "source_readiness_ref": {
    "path": "readiness.json",
    "checksum": "..."
  },
  "dataset_root": "data/SOL-ExecBench/benchmark",
  "selected_categories": ["L1"],
  "selection": {
    "status": "ready",
    "included_workloads": 1,
    "excluded_workloads": 2
  },
  "problems": [
    {
      "category": "L1",
      "problem_id": "L1/matmul_demo",
      "problem_path": "L1/matmul_demo",
      "definition_path": "L1/matmul_demo/definition.json",
      "workload_path": "L1/matmul_demo/workload.jsonl",
      "workloads": [{"uuid": "workload-1", "row_index": 1}]
    }
  ],
  "claim_boundary": {
    "ready_to_attempt_rocm_execution": true,
    "execution_success": false,
    "paper_level_validation": false,
    "hosted_leaderboard_parity": false,
    "upstream_solar_equivalence": false
  }
}
```

Planner should decide whether Phase 54 writes filtered workload JSONL files. The safer default is no: `ready_subset.json` should reference UUIDs/row indices and leave actual workload filtering to Phase 55, because Phase 54 is sidecar-only and does not execute. [VERIFIED: .planning/ROADMAP.md] [ASSUMED]

## CLI Strategy

Add `scripts/inspect_dataset.py` as a thin argparse wrapper, mirroring `scripts/download_solexecbench.py` rather than adding options to the primary `sol-execbench` CLI. [VERIFIED: scripts/download_solexecbench.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

Recommended CLI:

```bash
uv run scripts/inspect_dataset.py \
  --dataset-root data/SOL-ExecBench/benchmark \
  --manifest artifacts/dataset_manifest.json \
  --inventory artifacts/inventory.json \
  --readiness artifacts/readiness.json \
  --ready-subset artifacts/ready_subset.json
```

Use optional repeated `--category` if implementation wants parity with the downloader; route selection through `validate_categories()` for canonical ordering. [VERIFIED: src/sol_execbench/core/dataset/categories.py] [VERIFIED: scripts/download_solexecbench.py]

The script should return nonzero only for write/path/configuration failures, not because some problems are blocked. Blocked workloads are expected output data, not CLI failure. [VERIFIED: .planning/REQUIREMENTS.md] [ASSUMED]

## Standard Stack

No new package should be installed for Phase 54. The existing Python stack is sufficient. [VERIFIED: pyproject.toml] [VERIFIED: .planning/REQUIREMENTS.md]

| Library | Version / Constraint | Purpose | Why Standard |
|---------|----------------------|---------|--------------|
| Python | `>=3.12,<3.14` | Implementation runtime | Project baseline. [VERIFIED: pyproject.toml] |
| Pydantic | `>=2.12.5`, locked 2.12.5 | Sidecar and canonical schema models | Existing data and dataset contracts use Pydantic. [VERIFIED: pyproject.toml] [VERIFIED: uv.lock] |
| pytest | `>=9.0.2`, locked 9.0.2 | Fixture/unit tests | Existing test framework. [VERIFIED: pyproject.toml] [VERIFIED: uv.lock] |
| stdlib `json`, `pathlib`, `argparse` | Python stdlib | Deterministic serialization and thin scripts | Existing scripts and dataset helpers use these. [VERIFIED: scripts/download_solexecbench.py] [VERIFIED: src/sol_execbench/core/dataset/manifest.py] |

## Package Legitimacy Audit

No external packages are introduced by this phase, so the package legitimacy gate is not applicable. [VERIFIED: pyproject.toml]

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| none | — | — | — | — | — | No install planned. [VERIFIED: pyproject.toml] |

**Packages removed due to slopcheck [SLOP] verdict:** none. [VERIFIED: pyproject.toml]
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: pyproject.toml]

## Architecture Patterns

### System Architecture Diagram

```text
Dataset root + optional Phase 53 manifest
        |
        v
Category/problem discovery
        |
        v
Read definition.json + workload.jsonl files
        |
        +--> missing required file diagnostics + denominators
        |
        v
Parse with Definition / Workload Pydantic contracts
        |
        +--> schema_failure records + denominators
        |
        v
Inventory sidecar: problem records + workload child records
        |
        v
Static readiness classifier
        |
        +--> blockers, reason codes, evidence refs, next actions
        |
        v
Readiness sidecar
        |
        v
Ready-subset sidecar for Phase 55 execution selection
```

### Recommended Project Structure

```text
src/sol_execbench/core/dataset/
├── inventory.py       # Pydantic inventory models and builder
├── readiness.py       # status/reason-code models and classifier
├── ready_subset.py    # ready subset sidecar builder
└── __init__.py        # stable re-exports

scripts/
└── inspect_dataset.py # thin CLI wrapper

tests/sol_execbench/
└── test_dataset_inventory_readiness.py
```

This structure keeps Phase 54 in the dataset package and keeps scripts thin, matching the context decision. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md]

### Pattern 1: Deterministic Sidecar Serialization

Use `model_dump(mode="json")`, `json.dumps(..., indent=2, sort_keys=True) + "\n"`, and `stable_json_checksum()` with the checksum field nulled. [VERIFIED: src/sol_execbench/core/dataset/manifest.py] [VERIFIED: src/sol_execbench/core/dataset/checksums.py]

### Pattern 2: Structured Failure Records

Schema failures should be data records with stable `code`, `severity`, `category`, `problem_path`, optional `workload_uuid`/`row_index`, and `message`, following the `LayoutDiagnostic` style. [VERIFIED: src/sol_execbench/core/dataset/layout.py] [ASSUMED]

### Pattern 3: Public Contract Guardrails

Add assertions to `test_public_contract_guardrails.py` that inventory/readiness/ready-subset schema-version names do not appear in canonical `Definition`, `Workload`, or `Trace` dumps and that primary `sol-execbench --help` does not expose `--inventory`, `--readiness`, or `--ready-subset`. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dataset schema validation | Custom JSON key validators | `Definition` and `Workload` | They already encode public contract validation. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py] |
| Category validation/order | Local sorting or hard-coded duplicated sets | `validate_categories()` / `DEFAULT_CATEGORIES` | Existing helper owns canonical order. [VERIFIED: src/sol_execbench/core/dataset/categories.py] |
| Checksums | Ad hoc string concatenation | `stable_json_checksum()` | Existing helper produces stable SHA-256 over canonical JSON. [VERIFIED: src/sol_execbench/core/dataset/checksums.py] |
| Dataset execution | New runner | Phase 55 `scripts/run_dataset.py` path | Roadmap requires primary execution path reuse. [VERIFIED: .planning/ROADMAP.md] [VERIFIED: scripts/run_dataset.py] |
| Safetensors execution behavior | Static fake tensor substitution | Block or reference asset paths | Existing runtime loader validates real asset/key/shape/dtype. [VERIFIED: src/sol_execbench/core/bench/io.py] |

## Common Pitfalls

### Pitfall 1: Treating Readiness As Execution Success

**What goes wrong:** `ready` gets reported as passed/scored/paper parity. [VERIFIED: .planning/REQUIREMENTS.md]

**How to avoid:** Every readiness and ready-subset artifact should include claim-boundary booleans with execution and paper validation false, following Phase 53 manifest precedent. [VERIFIED: src/sol_execbench/core/dataset/manifest.py]

### Pitfall 2: Parsing With Raw JSON Only

**What goes wrong:** Inventory accepts malformed public schema that benchmark execution later rejects. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py]

**How to avoid:** Use raw JSON only to load text, then instantiate `Definition` and `Workload` for schema authority. [VERIFIED: .planning/REQUIREMENTS.md]

### Pitfall 3: Silent Random Substitution For Asset Inputs

**What goes wrong:** Missing custom/safetensors inputs appear runnable because inventory substitutes random tensors. [VERIFIED: src/sol_execbench/core/bench/io.py]

**How to avoid:** Classify missing or unresolved custom/safetensors requirements as blockers or explicit `needs_asset` evidence. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md]

### Pitfall 4: NVIDIA String False Positives

**What goes wrong:** Static CUDA/NVIDIA residue scanning blocks valid PyTorch ROCm compatibility names such as `torch.cuda`. [VERIFIED: docs/rocm_timing.md] [VERIFIED: tests/sol_execbench/test_rocm_eval_timing_audit.py]

**How to avoid:** Maintain an allowlist for documented compatibility namespace strings and only mark clear NVIDIA runtime/tooling hints as `unsupported_nvidia_only_path`. [VERIFIED: docs/rocm_timing.md] [ASSUMED]

### Pitfall 5: Modifying Canonical Dataset Files

**What goes wrong:** Inventory or ready-subset generation changes `definition.json`, `workload.jsonl`, `solution.json`, or trace JSONL. [VERIFIED: .planning/REQUIREMENTS.md]

**How to avoid:** All outputs are sidecar JSON artifacts; tests should compare canonical fixture files before/after generation. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] [ASSUMED]

## Tests And Fixtures

Add `tests/sol_execbench/test_dataset_inventory_readiness.py` with temporary dataset roots created under `tmp_path`, following the helper style in `test_dataset_contract.py`. [VERIFIED: tests/sol_execbench/test_dataset_contract.py]

Required fixture cases:

| Case | Assertions |
|------|------------|
| Schema-ok ready random workload | Inventory records dtypes/shapes/input kinds; readiness status `ready`; ready subset includes workload. [VERIFIED: src/sol_execbench/core/data/definition.py] |
| Definition schema failure | Inventory includes `schema_failure`; denominators count discovered but not parsed; run does not abort. [VERIFIED: .planning/REQUIREMENTS.md] |
| Workload schema failure | Problem parses; row diagnostic records `row_index`; readiness blocks that workload. [VERIFIED: src/sol_execbench/core/data/workload.py] |
| Missing required file | Missing `definition.json` or `workload.jsonl` appears in denominators and diagnostics. [VERIFIED: src/sol_execbench/core/dataset/layout.py] |
| Custom inputs | Workload with `custom` inputs is classified `custom_input_blocked` or explicit custom requirement, with no random substitution. [VERIFIED: src/sol_execbench/core/data/workload.py] |
| Safetensors asset | Existing path records asset requirement; missing path gets `runtime_blocked`/`safetensors_asset_missing`. [VERIFIED: src/sol_execbench/core/bench/io.py] |
| Unsupported dtype / Quant | FP8/FP4 or Quant records layered evidence and `needs_hardware_evidence` unless a stronger blocker exists. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: .planning/REQUIREMENTS.md] |
| NVIDIA-only hints | Static NVIDIA-only reference hint yields `unsupported_nvidia_only_path`; `torch.cuda` compatibility text is not enough by itself. [VERIFIED: docs/rocm_timing.md] [ASSUMED] |
| Determinism | Two runs with fixed timestamp produce identical JSON/checksums. [VERIFIED: tests/sol_execbench/test_dataset_contract.py] |
| Public guardrail | Canonical model dumps and primary CLI help remain unchanged. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |

## Code Examples

### Parse Problem Without Aborting Suite

```python
try:
    definition = Definition(**json.loads(definition_path.read_text(encoding="utf-8")))
except Exception as exc:
    return schema_failure_record("definition_schema_failure", definition_path, exc)
```

Source pattern: canonical `Definition` construction is used in tests and dataset scoring helpers. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] [VERIFIED: scripts/run_dataset.py]

### Deterministic JSON Output

```python
payload = artifact.model_dump(mode="json")
text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
```

Source pattern: `DatasetManifest.to_json()` uses this exact deterministic serialization style. [VERIFIED: src/sol_execbench/core/dataset/manifest.py]

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overblocking valid workloads | Ready subset too small | Emit reason codes and next actions so Phase 55/56 can refine blockers without mutating canonical files. [VERIFIED: .planning/ROADMAP.md] |
| Underblocking custom/safetensors workloads | Phase 55 attempts invalid local execution | Treat asset/evaluator requirements as blockers unless explicitly locatable. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] |
| Public schema drift | Breaks benchmark compatibility | Keep all new fields in sidecars and extend guardrail tests. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| Static NVIDIA scanning blocks compatibility APIs | False negative readiness | Use documented ROCm compatibility allowlist for `torch.cuda` namespace. [VERIFIED: docs/rocm_timing.md] |
| Low-precision readiness ambiguity | Misleading hardware claims | Model schema/input/reference/candidate/hardware layers separately and mark Quant/FP8/FP4 as needing evidence when otherwise ready. [VERIFIED: .planning/REQUIREMENTS.md] |

## Plan-Shaping Recommendation

Split Phase 54 into three implementation plans:

1. **Inventory core and fixtures:** Add `inventory.py`, deterministic models/checksum helpers, denominator accounting, schema-failure diagnostics, and inventory tests for INV-01..INV-05. [VERIFIED: .planning/REQUIREMENTS.md]
2. **Readiness classifier:** Add `readiness.py` with status enum, reason-code vocabulary, layered evidence fields, custom/safetensors/low-precision/NVIDIA-only classification, and tests for READY-01..READY-04. [VERIFIED: .planning/REQUIREMENTS.md]
3. **Ready subset and CLI/guardrails:** Add `ready_subset.py`, `scripts/inspect_dataset.py`, `__init__.py` exports, public guardrail tests, and docs wording assertions for READY-05 and claim boundaries. [VERIFIED: .planning/ROADMAP.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

This split keeps the parser, classifier, and CLI side effects separable and makes each plan independently testable without GPU or network access. [VERIFIED: AGENTS.md] [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Local tests/scripts | yes | 3.12.13 | None needed. [VERIFIED: python3 --version] |
| uv | Running tests/scripts | yes | 0.11.15 | Direct `python -m pytest` if environment is already synced. [VERIFIED: uv --version] |
| pytest | Validation | configured | 9.0.2 locked | None needed. [VERIFIED: pyproject.toml] [VERIFIED: uv.lock] |
| ROCm GPU | Phase 54 | not required | — | Fixture-only tests. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] |
| Network / Hugging Face | Phase 54 | not required | — | Use local tmp fixtures. [VERIFIED: tests/sol_execbench/test_download_solexecbench.py] |

**Missing dependencies with no fallback:** none. [VERIFIED: pyproject.toml]

**Missing dependencies with fallback:** none for Phase 54. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 with xdist. [VERIFIED: pyproject.toml] [VERIFIED: uv.lock] |
| Config file | `pyproject.toml`. [VERIFIED: pyproject.toml] |
| Quick run command | `uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` [ASSUMED] |
| Full suite command | `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` [ASSUMED] |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| INV-01 | Parses every discovered problem/workload through Pydantic contracts. | unit | `uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -n 0` | no; Wave 0. [ASSUMED] |
| INV-02 | Records problem/workload metadata, dtypes, input kinds, assets, references, solutions. | unit | same | no; Wave 0. [ASSUMED] |
| INV-03 | Emits conservative op/direction hints with unknowns. | unit | same | no; Wave 0. [ASSUMED] |
| INV-04 | Emits category/suite denominators. | unit | same | no; Wave 0. [ASSUMED] |
| INV-05 | Deterministic repeated output. | unit | same | no; Wave 0. [ASSUMED] |
| READY-01 | Classifies required statuses. | unit | same | no; Wave 0. [ASSUMED] |
| READY-02 | Emits reason codes/evidence paths/next actions. | unit | same | no; Wave 0. [ASSUMED] |
| READY-03 | Blocks explicit custom/safetensors requirements. | unit | same | no; Wave 0. [ASSUMED] |
| READY-04 | Emits layered low-precision/Quant evidence. | unit | same | no; Wave 0. [ASSUMED] |
| READY-05 | Builds sidecar ready subset only. | unit | same | no; Wave 0. [ASSUMED] |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/sol_execbench/test_dataset_inventory_readiness.py -n 0` [ASSUMED]
- **Per wave merge:** `uv run pytest tests/sol_execbench/test_dataset_contract.py tests/sol_execbench/test_download_solexecbench.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` [ASSUMED]
- **Phase gate:** Full targeted suite green before `$gsd-verify-work`. [VERIFIED: AGENTS.md]

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_dataset_inventory_readiness.py` covers INV-01..INV-05 and READY-01..READY-05. [ASSUMED]
- [ ] Guardrail additions in `tests/sol_execbench/test_public_contract_guardrails.py` for sidecar-only fields and no primary CLI exposure. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- [ ] `scripts/inspect_dataset.py` import test or direct `main(argv)` test, matching downloader tests. [VERIFIED: tests/sol_execbench/test_download_solexecbench.py]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No auth surface. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] |
| V3 Session Management | no | No session surface. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] |
| V4 Access Control | yes | Treat artifact paths as explicit local paths; do not traverse outside caller-selected output paths. [VERIFIED: scripts/download_solexecbench.py] [ASSUMED] |
| V5 Input Validation | yes | Use Pydantic `Definition` and `Workload` contracts plus safe category validation. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py] [VERIFIED: src/sol_execbench/core/dataset/categories.py] |
| V6 Cryptography | yes | Use SHA-256 checksums via existing helper; do not hand-roll crypto. [VERIFIED: src/sol_execbench/core/dataset/checksums.py] |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via problem names or artifact refs | Tampering | Use relative paths from dataset root and reject unsafe output targets where needed; downloader already rejects unsafe remote problem names. [VERIFIED: scripts/download_solexecbench.py] |
| Canonical dataset mutation | Tampering | Sidecar-only writes; tests assert canonical files unchanged. [VERIFIED: .planning/REQUIREMENTS.md] [ASSUMED] |
| Misleading validation claims | Repudiation / Information disclosure | Claim-boundary fields and wording tests. [VERIFIED: src/sol_execbench/core/dataset/manifest.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| Execution of untrusted reference code during inventory | Elevation of privilege | Do not import or execute references in Phase 54; parse as schema text only. [VERIFIED: .planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md] |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Exact new schema-version strings should be `sol_execbench.dataset_inventory.v1`, `sol_execbench.rocm_readiness.v1`, and `sol_execbench.ready_subset.v1`. | Inventory Model Proposal | Naming churn if planner prefers different stable names. |
| A2 | Workload-row parse failures should be represented as child records instead of problem-level fatal records. | Inventory Model Proposal | Denominator design may need adjustment. |
| A3 | Fixed readiness severity order is the right tie-breaker. | Readiness Status And Reason-Code Proposal | Problem summaries may differ from user expectations. |
| A4 | Ready subset should reference UUID/row index and not materialize filtered workload files in Phase 54. | Ready-Subset Shape | Phase 55 may need an extra filtering step. |
| A5 | `scripts/inspect_dataset.py` should return zero when blockers are successfully reported. | CLI Strategy | CI scripts may expect nonzero for blocked readiness. |
| A6 | NVIDIA-only static detection can be based on a compatibility allowlist plus clear runtime/tooling hints. | Common Pitfalls / Tests | False positives or false negatives may require refining reason codes. |
| A7 | Three implementation plans are the best split. | Plan-Shaping Recommendation | Planner may merge or split differently based on wave sizing. |

## Open Questions (RESOLVED)

1. **Should ready-subset materialize filtered workload files in Phase 54?**
   - What we know: Phase 54 is sidecar-only and Phase 55 owns execution. [VERIFIED: .planning/ROADMAP.md]
   - What's unclear: Whether Phase 55 runner changes are simpler with prefiltered JSONL files.
   - Resolution: Do not materialize filtered workload files in Phase 54. The
     ready-subset artifact will reference problem paths and workload UUID/row
     indices only; Phase 55 owns execution-time filtering. [ASSUMED]

2. **How aggressive should NVIDIA-only static detection be?**
   - What we know: PyTorch ROCm legitimately exposes HIP devices through `torch.cuda`. [VERIFIED: docs/rocm_timing.md]
   - What's unclear: The public dataset may contain CUDA strings that are comments, compatibility names, or true blockers.
   - Resolution: Use explicit reason codes and a conservative allowlist. Do not
     block solely on documented compatibility strings such as `torch.cuda`;
     reserve `unsupported_nvidia_only_path` for clear CUDA/NVIDIA runtime or
     tooling dependency hints. [ASSUMED]

3. **Should safetensors tensor keys be statically verified?**
   - What we know: Runtime loader verifies keys, shapes, and dtypes when loading files. [VERIFIED: src/sol_execbench/core/bench/io.py]
   - What's unclear: Phase 54 may not have all blob roots or may not want to open large assets.
   - Resolution: Verify safetensors path existence only by default. Record the
     tensor key as required evidence; classify missing files as
     `runtime_blocked` and leave key/shape/dtype runtime validation to later
     execution. [ASSUMED]

## Sources

### Primary (HIGH confidence)

- `AGENTS.md` - project constraints, testing rules, style, and ROCm scope.
- `.planning/phases/54-paper-inventory-and-rocm-readiness-classification/54-CONTEXT.md` - locked phase decisions.
- `.planning/REQUIREMENTS.md` - INV/READY requirement vocabulary.
- `.planning/ROADMAP.md` - Phase 54/55/56/57 boundaries.
- `.planning/STATE.md` - current milestone state and deferred claim boundaries.
- `src/sol_execbench/core/dataset/*.py` - Phase 53 dataset layout/manifest patterns.
- `src/sol_execbench/core/data/definition.py` - canonical definition contract.
- `src/sol_execbench/core/data/workload.py` - canonical workload contract.
- `src/sol_execbench/core/bench/io.py` - input generation and safetensors behavior.
- `scripts/run_dataset.py` - existing execution path and Phase 55 target.
- `scripts/download_solexecbench.py` - thin script and manifest pattern.
- `tests/sol_execbench/test_dataset_contract.py` - dataset sidecar test style.
- `tests/sol_execbench/test_download_solexecbench.py` - script test style and safety checks.
- `tests/sol_execbench/test_public_contract_guardrails.py` - public schema and CLI guardrail pattern.
- `docs/rocm_timing.md` - ROCm compatibility naming and derived evidence boundary.

### Secondary (MEDIUM confidence)

- `pyproject.toml` and `uv.lock` - local dependency constraints and locked versions.
- `tests/conftest.py` - hardware marker behavior.
- `docs/internal/v1_4_compatibility_inventory.md` - historical public contract inventory.

### Tertiary (LOW confidence)

- None. No external web search was needed because the phase is constrained to existing repository contracts and no new dependency is proposed.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - no new packages; existing stack verified in project files.
- Architecture: HIGH - phase context and existing dataset package establish module boundary.
- Schema details: HIGH - verified against current Pydantic model source and guardrail tests.
- Readiness reason codes: MEDIUM - status vocabulary is required, exact reason-code names are proposed.
- CLI strategy: HIGH - context explicitly requires a thin script and no primary CLI change.
- Tests/fixtures: HIGH - existing test style and required fixture cases are clear.

**Research date:** 2026-05-23
**Valid until:** 2026-06-22

## RESEARCH COMPLETE
