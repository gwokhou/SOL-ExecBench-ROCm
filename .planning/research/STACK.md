# Technology Stack

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.9 AMD SOL/SOLAR Bound Modeling Completion
**Researched:** 2026-05-22
**Scope:** RDNA 4 bound-modeling completion only. CDNA 3 / MI300X and CDNA 4 validation remain deferred.
**Overall confidence:** HIGH for repository integration points; MEDIUM for final operator formulas until phase-specific golden cases are selected.

## Recommendation

Do not add a new framework dependency for v1.9. The right stack is the existing Python 3.12 stdlib + Pydantic v2 + frozen dataclass artifact pattern already used by `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_score.py`, and `src/sol_execbench/core/scoring/baseline_artifact.py`.

The structural change should be an internal scorer/analyzer split:

1. Keep canonical trace JSONL unchanged.
2. Replace the current single-file AST-only estimator path with a small local analyzer package under `src/sol_execbench/core/scoring/amd_sol/` or closely scoped sibling modules.
3. Add a typed local operation IR that records source expressions, tensor roles, shape formulas, operation formulas, confidence, and unsupported reasons.
4. Externalize AMD hardware model inputs as versioned JSON artifacts loaded with stdlib `json` and validated with existing typed model patterns.
5. Emit derived AMD SOL bound artifacts v2 and feed them into the existing AMD-native score report path through `score_amd_native_trace_workload()` and `scripts/run_dataset.py --amd-score-report`.

## Recommended Stack Additions

### Core Python Libraries

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python stdlib `ast` | Python 3.12+ | Parse `Definition.reference` into a source-level operation graph | Already used in `Definition` validation and current AMD SOL extraction; sufficient for local Python reference code without parser dependencies. |
| Python stdlib `json` | Python 3.12+ | Load hardware model and bound artifact JSON | Matches `baseline_artifact.py`; avoids YAML/TOML dependency churn and keeps artifacts easy to diff. |
| Python stdlib `dataclasses` | Python 3.12+ | Internal IR and derived artifact objects | Current scoring modules use frozen dataclasses with `to_dict()`; preserve the local style for non-public derived evidence. |
| Pydantic v2 | Existing project dependency | Validate public or semi-public artifact loaders where stronger input validation is needed | Already used for canonical data schemas. Use it for hardware model artifact validation if loader errors must be schema-like; do not introduce Marshmallow, attrs, or msgspec. |
| Pytest | Existing dev dependency | Golden operator/model/report tests | Existing tests already cover `amd_sol`, `amd_score`, dataset integration, and public guardrails. |
| Ruff | Existing dev dependency | Style and formatting | Required by repository conventions. |

### Local Analyzer Modules

| Module | Responsibility | Notes |
|--------|----------------|-------|
| `core/scoring/amd_sol_ir.py` or `core/scoring/amd_sol/ir.py` | Define `SolOpNode`, tensor refs, formulas, memory roles, and confidence enums | Prefer frozen dataclasses. Keep enum values compatible with existing `supported`, `inexact`, `unsupported` labels. |
| `core/scoring/amd_sol_parser.py` or `core/scoring/amd_sol/parser.py` | Convert `Definition.reference` AST into IR nodes | Keep this AST-based. Add explicit handlers for call chains, returns, tuple outputs, `torch.*`, `F.*`, tensor methods, and binary operators. |
| `core/scoring/amd_sol_formulas.py` or `core/scoring/amd_sol/formulas.py` | Compute FLOPs, element counts, bytes, and pass counts from `Definition`, `Workload`, and IR | Move current `estimate_work()` heuristics here and make each operator formula auditable. |
| `core/scoring/amd_hardware.py` or `core/scoring/amd_sol/hardware.py` | Load and validate hardware model JSON artifacts | Replaces `default_amd_hardware_models()` as the primary path while keeping it as a compatibility fallback if needed. |
| `core/scoring/amd_sol.py` | Public compatibility facade | Preserve imports such as `build_amd_sol_bound_artifact()` and `default_amd_hardware_models()` to reduce downstream churn. |

Either a package directory or sibling modules is acceptable. A package directory is cleaner if v1.9 adds more than two new files; a facade file keeps existing imports stable.

### Data Artifacts

| Artifact | Format | Schema Version | Purpose | Recommendation |
|----------|--------|----------------|---------|----------------|
| AMD hardware model | JSON | `sol_execbench.amd_hardware_model.v1` | Architecture, dtype/path peaks, memory bandwidth, clock policy, source, validation status, and notes | Add `data/amd_hardware_models/gfx1200.json` or `src/sol_execbench/data/amd_hardware_models/gfx1200.json` depending on whether it is packaged. For runtime defaults, package it under `src/sol_execbench/data/` and load via `importlib.resources`. |
| AMD SOL bound artifact | JSON dict emitted from dataclasses | `sol_execbench.amd_sol_bound.v2` | Per-workload graph, work, memory movement, hardware model ref, aggregate bound, and coverage | Version bump is warranted because v1.9 changes artifact semantics from shallow graph nodes to structured IR and explicit memory evidence. |
| AMD-native score report | JSON dict emitted from dataclasses | Keep `sol_execbench.amd_native_score.v1` unless fields change | Suite/workload score output | Prefer reusing existing `evidence_refs` and warnings. Bump only if the score JSON shape changes. |
| Baseline artifact | Existing JSON | Keep `sol_execbench.scoring_baseline.v1` | Release-defined baseline timing | No change needed for bound modeling. |

Hardware model JSON should include enough provenance for audits:

```json
{
  "schema_version": "sol_execbench.amd_hardware_model.v1",
  "architecture": "gfx1200",
  "model_name": "RDNA 4 validation target",
  "dtype_paths": [
    {
      "dtype_or_path": "bf16/fp32 mixed benchmark path",
      "peak_tflops": 48.0,
      "memory_bandwidth_gbps": 640.0,
      "source": "project provisional RDNA4 model input; validate before publication",
      "confidence": "inexact",
      "validation_status": "provisional"
    }
  ],
  "clock_policy": "documented benchmark clock policy",
  "validation_scope": "RDNA 4 only"
}
```

Keep numbers provisional unless backed by recorded RDNA 4 evidence. Do not add MI300X/CDNA 3 artifacts as validated inputs in v1.9.

## IR Requirements

The local IR should be purpose-built, not a general graph framework. Each node should record:

| Field | Why |
|-------|-----|
| `node_id` | Stable evidence refs in artifacts and tests. |
| `op_type` | Operator family: `matmul`, `elementwise`, `reduction`, `normalization`, `softmax`, `data_movement`, `unsupported`. |
| `source_expression` | Auditable link back to reference code. |
| `inputs` / `outputs` | Needed for byte accounting and multi-output references. |
| `shape_exprs` / resolved shapes | Needed for workload-specific FLOP and byte formulas. |
| `formula` | Human-readable formula such as `2*M*N*K` or `5*numel`. |
| `memory_reads` / `memory_writes` | Explicit movement evidence instead of one global tensor byte estimate. |
| `confidence` and `rationale` | Existing score guardrails depend on supported/inexact/unsupported degradation. |

Do not use `networkx`, `sympy`, `torch.fx`, ONNX, MLIR, or Triton parser infrastructure for v1.9. They add dependency and semantic overhead without matching the repository's current source of truth: small Python reference functions embedded in `definition.json`.

## Integration Points

| Existing File | v1.9 Change |
|---------------|-------------|
| `src/sol_execbench/core/scoring/amd_sol.py` | Convert to facade or split internals while preserving `build_amd_sol_bound_artifact()`. Emit v2 artifacts with explicit formula and memory evidence. |
| `src/sol_execbench/core/scoring/amd_score.py` | Keep score formula path. Add warnings only if v2 introduces new confidence states or hardware-model validation outcomes. Preserve `evidence_refs`. |
| `src/sol_execbench/core/scoring/baseline_artifact.py` | No structural change. Use as the model for simple JSON loader style. |
| `src/sol_execbench/core/data/definition.py` | No public schema change. Reuse `Definition.get_resolved_axes_values()`, input/output shape helpers, and existing AST validation. |
| `scripts/run_dataset.py` | Add optional hardware model artifact path only if needed. Otherwise default to packaged RDNA 4 model and keep `--amd-score-report` as the integration point. |
| `docs/analysis.md` | Document v2 bound artifact shape, hardware model provenance, unsupported/inexact degradation, and RDNA 4-only validation scope. |

Avoid adding fields to canonical trace JSONL. All new bound-modeling evidence belongs in derived artifacts and score reports.

## Tests

Add focused tests rather than broad integration churn:

| Test Area | Location | Required Coverage |
|-----------|----------|-------------------|
| Parser/IR golden cases | `tests/sol_execbench/test_amd_sol_bounds.py` or new `test_amd_sol_ir.py` | Matmul, matmul epilogue, elementwise chains, reductions, softmax, normalization, views/reshapes, tuple outputs, unsupported calls. |
| Formula accounting | `tests/sol_execbench/test_amd_sol_bounds.py` | FLOP formulas, read/write bytes, aggregate bound, limiting resource, confidence summaries. |
| Hardware model loader | New or existing scoring test | Valid RDNA 4 provisional artifact, invalid schema, non-positive peak/bandwidth rejection, CDNA 3 unvalidated warning preservation. |
| Score integration | `tests/sol_execbench/test_amd_native_score.py` | v2 artifact still scores through existing report path; unsupported/inexact warnings remain mandatory. |
| Dataset integration | `tests/sol_execbench/test_run_dataset_amd_score.py` | Packaged/default hardware model evidence refs survive report generation. |
| Public guardrails | `tests/sol_execbench/test_public_contract_guardrails.py` | Canonical trace payload unchanged; no NVIDIA/B200/SOLAR leaderboard claim language. |
| Docs guardrails | Existing docs tests if present | `docs/analysis.md` says derived artifacts only and RDNA 4 validation only. |

Recommended validation commands:

```bash
uv run pytest tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_public_contract_guardrails.py
uv run ruff check src/sol_execbench/core/scoring tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py
```

## What Not To Add

| Do Not Add | Reason |
|------------|--------|
| `networkx` | The needed graph is small, linear-ish, and evidence-oriented; dataclasses are enough. |
| `sympy` | Shape formulas already resolve through project helpers; symbolic algebra would expand scope and dependency surface. |
| `pandas` / `polars` | JSON artifacts and pytest assertions do not require dataframe tooling. |
| YAML/TOML config dependencies | JSON is already used for trace, baseline, solution, and artifact contracts. |
| `torch.fx` / ONNX export | The source of truth is reference Python in `Definition.reference`, not an executed PyTorch module graph. FX also risks runtime/device side effects. |
| New canonical trace fields | Violates the milestone quality gate; derived bound and score artifacts are the correct layer. |
| CDNA 3 / MI300X validated hardware models | Explicitly out of v1.9 scope. Keep only unvalidated/provisional scaffolding if needed for guardrail tests. |
| NVIDIA SOLAR/B200 compatibility layer | The project must avoid leaderboard-equivalence claims. AMD-native score reports remain derived ROCm interpretation artifacts. |

## Sources

- `.planning/PROJECT.md` - v1.9 scope, RDNA 4 validation boundary, deferred CDNA 3/CDNA 4 work.
- `src/sol_execbench/core/scoring/amd_sol.py` - current AST graph extraction, work estimates, hardware defaults, bound artifact shape.
- `src/sol_execbench/core/scoring/amd_score.py` - AMD-native score integration, warning behavior, evidence refs.
- `src/sol_execbench/core/scoring/baseline_artifact.py` - local JSON artifact loader pattern.
- `src/sol_execbench/core/data/definition.py` - canonical definition schema, AST validation, shape/dtype source.
- `docs/analysis.md` - trace immutability, AMD-native score interpretation, coverage semantics.
- `tests/sol_execbench/test_amd_sol_bounds.py`, `tests/sol_execbench/test_amd_native_score.py`, `tests/sol_execbench/test_run_dataset_amd_score.py`, `tests/sol_execbench/test_public_contract_guardrails.py` - existing coverage and guardrail locations.
