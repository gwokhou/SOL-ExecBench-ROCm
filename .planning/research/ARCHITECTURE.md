# Architecture Research: AMD SOL/SOLAR Bound Modeling Completion

**Project:** SOL ExecBench ROCm Port v1.9
**Domain:** Derived AMD SOL/SOLAR bound modeling for ROCm benchmark scoring
**Researched:** 2026-05-22
**Overall confidence:** HIGH for integration boundaries, MEDIUM for exact model formulas until operator golden cases are specified.

## Executive Summary

The new bound-modeling work should integrate as a derived scoring subsystem under `src/sol_execbench/core/scoring/`, not as part of the benchmark execution path. The public benchmark contract remains `definition.json`, `workload.jsonl`, `solution.json`, and canonical `Trace` JSONL from `src/sol_execbench/core/data/trace.py`. AMD SOL/SOLAR artifacts are sidecar evidence consumed by AMD-native score reports, and they must never be embedded into, backfilled into, or required for canonical trace emission.

The existing `amd_sol.py` already proves the right boundary: it builds `AmdSolBoundArtifact` from a `Definition`, `Workload`, and `AmdHardwareModel`, while `amd_score.py` consumes that artifact plus traces and baselines to produce guarded `AmdNativeScore` / `AmdNativeSuiteReport`. v1.9 should keep that external API shape but split the internals into focused modules for IR extraction, operator estimation, hardware-model loading, artifact serialization, and coverage/confidence reporting.

The highest-value architectural move is replacing the current AST visitor plus hard-coded hardware defaults with a small, typed pipeline:

`Definition.reference + Workload -> BoundGraph -> WorkEstimate[] -> HardwareModel -> BoundArtifact -> AmdNativeScore`.

The dataset runner should be modified only at the optional derived-report layer. `scripts/run_dataset.py --amd-score-report` can load hardware model artifacts, emit per-workload SOL bound sidecars, and reference them from the suite report. The benchmark CLI and `Trace` schema should not change.

## Public-Contract Boundaries

| Boundary | Status | Rule |
| --- | --- | --- |
| `Trace` JSONL in `src/sol_execbench/core/data/trace.py` | Unchanged | Do not add SOL fields, schema versions, coverage, or hardware models to canonical traces. |
| Primary `sol-execbench` CLI | Unchanged by default | It should continue to emit trace JSONL only. No SOL artifact generation in the hot benchmark path unless a later explicit CLI flag is designed. |
| `definition.json` / `workload.jsonl` / `solution.json` | Unchanged | Bound modeling reads validated models and must not require new public schema fields. |
| `AmdNativeSuiteReport` | Modified derived contract | It may gain evidence references and scoring warnings, but remains a derived report with `canonical_output: trace_jsonl`. |
| AMD SOL bound artifacts | New/modified derived contract | Version independently from public benchmark schemas. Store as JSON sidecars with explicit `schema_version`, `derived: true`, model source, confidence, and coverage. |
| Hardware model artifacts | New derived input contract | Version independently and load explicitly; built-in fallback may exist but must be labeled provisional. |

## Recommended Module Structure

### New Components

| Module | Classes / Functions | Responsibility |
| --- | --- | --- |
| `src/sol_execbench/core/scoring/amd_sol/ir.py` | `BoundGraph`, `BoundNode`, `BoundEdge`, `TensorRef`, `OpKind`, `GraphConfidence` | Typed graph/IR extracted from reference code and resolved workload shapes. |
| `src/sol_execbench/core/scoring/amd_sol/extract.py` | `extract_bound_graph(definition, workload)` | Convert `Definition.reference` plus workload axes into a normalized operation graph. This replaces the current private `_GraphVisitor` as the public internal entry point. |
| `src/sol_execbench/core/scoring/amd_sol/estimates.py` | `WorkEstimate`, `MemoryMovementEstimate`, `estimate_work(graph, definition, workload)` | Produce auditable FLOP, byte, read/write, and movement estimates per node. |
| `src/sol_execbench/core/scoring/amd_sol/hardware.py` | `AmdHardwareModel`, `HardwareValidationStatus`, `load_amd_hardware_models(path)`, `select_hardware_model(models, arch, dtype_or_path)` | Load versioned hardware models from JSON artifacts and keep provisional defaults isolated. |
| `src/sol_execbench/core/scoring/amd_sol/artifact.py` | `AmdSolBoundArtifact`, `AmdSolCoverageSummary`, `build_amd_sol_bound_artifact(...)`, `load_amd_sol_bound_artifact(path)` | Own artifact schema versioning, serialization, loading, and backward-compatible construction. |
| `src/sol_execbench/core/scoring/amd_sol/operators.py` | operator estimator registry | Encapsulate formulas for matmul, elementwise, reductions, normalization, softmax, transpose/view/data movement, and unsupported fallback. |
| `src/sol_execbench/core/scoring/amd_sol/coverage.py` | `summarize_coverage(...)` | Aggregate supported/inexact/unsupported counts and operation-family coverage. |
| `src/sol_execbench/core/scoring/amd_sol/schema.py` | constants and migration helpers | Define `sol_execbench.amd_sol_bound.v2` and hardware model schema versions. |

Keep `src/sol_execbench/core/scoring/amd_sol.py` temporarily as a compatibility facade if avoiding import churn matters. It can re-export the new classes and functions, then later be retired after internal imports move to the package.

### Modified Components

| Component | Modification |
| --- | --- |
| `src/sol_execbench/core/scoring/amd_score.py` | Consume v2 artifacts without needing to know extraction details. Keep `score_amd_native_trace_workload(trace, artifact, ...)` as the core integration function. Add warnings for missing artifact version, provisional hardware, inexact coverage, and unsupported op families. |
| `scripts/run_dataset.py` | Add optional flags for hardware model path and SOL artifact output directory. Generate sidecars only when `--amd-score-report` or explicit SOL output is requested. Reference sidecar paths in `evidence_refs["sol_bound"]` and `evidence_refs["hardware_model"]`. |
| `docs/analysis.md` | Document that AMD-native score reports require derived SOL bound artifacts and that unsupported/inexact coverage limits claims. |
| `docs/ARCHITECTURE.md` | Add a short derived-scoring data-flow section showing trace JSONL remains canonical and SOL artifacts are sidecars. |
| `docs/rocm.md` or a new `docs/amd_sol_bounds.md` | Document hardware model artifacts, validation status, RDNA 4 scope, and CDNA 3/CDNA 4 deferral. |
| `tests/sol_execbench/test_amd_sol_bounds.py` | Move or expand into focused tests for IR extraction, formulas, artifact loading, versioning, and no-trace-mutation guardrails. |
| `tests/sol_execbench/test_run_dataset_amd_score.py` | Add artifact-output and hardware-model-loading coverage for dataset report integration. |

### Unchanged Components

| Component | Why unchanged |
| --- | --- |
| `src/sol_execbench/core/data/trace.py` | Canonical trace schema is the primary public output contract and already has a guardrail test proving SOL artifacts do not mutate traces. |
| `src/sol_execbench/driver/templates/eval_driver.py` | Bound modeling derives from definitions, workloads, traces, baselines, and hardware models after execution. The evaluation subprocess should not need scoring knowledge. |
| `src/sol_execbench/driver/problem_packager.py` | No staging change is needed because bound artifacts are post-processing outputs, not candidate execution inputs. |
| `src/sol_execbench/cli/main.py` | Keep primary benchmark behavior stable. Add a separate CLI later only if there is a clear user workflow; do not overload trace emission. |

## Recommended Data Flow

### Benchmark Path, Unchanged

```text
definition.json + workload.jsonl + solution.json
  -> ProblemPackager
  -> build_ext.py if native ROCm
  -> eval_driver.py subprocess
  -> canonical Trace JSONL
```

### Derived Bound And Score Path

```text
Definition + Workload
  -> extract_bound_graph()
  -> estimate_work()
  -> load/select AmdHardwareModel
  -> build AmdSolBoundArtifact JSON sidecar

Trace JSONL + AmdSolBoundArtifact + optional ScoringBaselineArtifact
  -> score_amd_native_trace_workload()
  -> AmdNativeSuiteReport JSON
```

The artifact reference key should be stable:

```text
(definition_name, workload_uuid, hardware_model_id, artifact_schema_version)
```

Using only `workload_uuid` is convenient but too weak once the dataset runner may process multiple definitions or architectures in one output tree.

## Artifact Loading And Storage

### Hardware Model Artifacts

Store hardware model inputs as explicit JSON, for example:

```text
data/amd_hardware_models/gfx1200.json
```

Recommended fields:

```json
{
  "schema_version": "sol_execbench.amd_hardware_model.v1",
  "architecture": "gfx1200",
  "model_id": "gfx1200-rdna4-v1",
  "dtype_or_path": "fp32/bf16 benchmark path",
  "peak_tflops": 48.0,
  "memory_bandwidth_gbps": 640.0,
  "source": "project RDNA4 validation input",
  "validation_status": "validated",
  "confidence": "supported",
  "validation_scope": "RDNA 4 only",
  "notes": []
}
```

`default_amd_hardware_models()` can remain as a fallback, but it should call the same parser used for JSON artifacts and mark any fallback as `provisional` or `unvalidated`. Do not silently treat built-ins as release-grade validation.

### SOL Bound Artifacts

Store one artifact per workload or one JSON report with a top-level artifact array. Per-workload JSON is easier to diff and reference:

```text
out/amd_sol_bounds/<category>/<problem>/<workload_uuid>.amd_sol_bound.json
```

The artifact should include:

- `schema_version` such as `sol_execbench.amd_sol_bound.v2`.
- `derived: true`.
- `definition`, `workload_uuid`, and optional `workload_axes`.
- `hardware_model_ref` and embedded minimal hardware summary.
- `graph` with nodes, op kinds, tensor refs, and confidence.
- `work_estimates` with FLOPs, bytes read, bytes written, total bytes, memory movement category, confidence, and formula rationale.
- `op_bounds` with compute, memory, aggregate bound, limiting resource, and confidence.
- `coverage_summary` with supported/inexact/unsupported counts by op family.
- `warnings` suitable for report propagation.

Avoid embedding full trace payloads in SOL artifacts. Use references only.

## Schema Versioning

Use independent versions:

| Schema | Current / Proposed | Owner |
| --- | --- | --- |
| Canonical trace | Existing public model, no v1.9 change | `core/data/trace.py` |
| AMD SOL bound artifact | `sol_execbench.amd_sol_bound.v2` | `core/scoring/amd_sol/artifact.py` |
| AMD hardware model | `sol_execbench.amd_hardware_model.v1` | `core/scoring/amd_sol/hardware.py` |
| AMD-native score report | Existing `sol_execbench.amd_native_score.v1`, only bump if JSON shape changes incompatibly | `core/scoring/amd_score.py` |

Prefer additive changes for score reports. If `AmdNativeSuiteReport.to_dict()` gains only optional evidence or warning fields, keep v1. If required fields or interpretation changes, bump to v2 and provide loader tests.

## Test Layout

| Test File | Coverage |
| --- | --- |
| `tests/sol_execbench/core/scoring/test_amd_sol_ir.py` | AST/reference extraction into typed graph nodes for matmul, elementwise, reductions, normalization, softmax, data movement, and unsupported calls. |
| `tests/sol_execbench/core/scoring/test_amd_sol_estimates.py` | Golden FLOP/byte/memory-movement estimates for common SOL ExecBench families. |
| `tests/sol_execbench/core/scoring/test_amd_hardware_models.py` | JSON load/validation, provisional fallback, RDNA 4 validation metadata, CDNA 3/CDNA 4 no-claim guardrails. |
| `tests/sol_execbench/core/scoring/test_amd_sol_artifacts.py` | Artifact `to_dict()` / loader round trips, schema version checks, evidence refs, coverage summaries. |
| `tests/sol_execbench/test_amd_native_score.py` | Score integration warnings and no mutation of `Trace`. |
| `tests/sol_execbench/test_run_dataset_amd_score.py` | Dataset runner writes sidecars, loads hardware models, links evidence refs, and leaves trace files unchanged. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Assert trace docs/schema and primary CLI output stay free of AMD SOL artifact fields. |
| `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` | Claim-level warnings for unsupported/inexact/provisional/CDNA-deferred evidence. |

Golden tests should use tiny synthetic definitions first, then a small set of existing samples such as matmul, RMSNorm, softmax-like, and linear backward. RDNA 4 validation can be an integration gate for artifact generation plus report production, but most formula tests should be hardware-free.

## Documentation Plan

| Document | Change |
| --- | --- |
| `docs/ARCHITECTURE.md` | Add derived AMD SOL/SOLAR scoring sidecar flow and explicit public-contract boundary. |
| `docs/analysis.md` | Explain score prerequisites: canonical trace, scoring baseline or fallback, SOL bound artifact, hardware model, and timing evidence. |
| `docs/amd_sol_bounds.md` | New focused doc for artifact schemas, hardware model fields, confidence levels, supported op families, and RDNA 4 validation scope. |
| `docs/internal/mi300x_validation_readiness.md` | Keep CDNA 3 / MI300X validation deferred; mention that v1.9 only prepares model slots and no hardware claim. |
| `README.md` or `docs/GETTING-STARTED.md` | Only add a short pointer to derived AMD-native scoring if user-facing workflow changes. |

## Suggested Build Order

1. **Contract guardrails first** - Add tests asserting canonical `Trace` JSONL, primary CLI JSON output, and public data schemas remain unchanged. This prevents accidental scoring data leakage into benchmark output.
2. **Hardware model loader** - Implement versioned hardware model parsing and RDNA 4 artifact fixtures before formula work, because every bound artifact needs a validated or provisional model source.
3. **IR extraction package** - Split the current AST visitor into typed graph extraction with stable node IDs, op kinds, tensor/shape context, and unsupported-node preservation.
4. **Operator estimators** - Add golden estimates for matmul, elementwise, reduction, normalization/RMSNorm, softmax, transpose/view/data movement, and unsupported fallback. Prefer explicit inexact confidence over silent precision.
5. **Bound artifact v2** - Build/load JSON sidecars from graph, estimates, hardware models, and coverage summaries. Keep a facade so existing `build_amd_sol_bound_artifact()` callers continue working.
6. **Score integration** - Update `amd_score.py` to consume v2 artifacts and propagate coverage/hardware warnings while preserving `Trace` immutability.
7. **Dataset runner integration** - Add optional sidecar output and hardware model flags to `scripts/run_dataset.py`; wire evidence refs into `--amd-score-report`.
8. **Docs and claim guardrails** - Document artifact schema, RDNA 4 scope, CDNA deferral, and unsupported/inexact degradation behavior.
9. **RDNA 4 validation pass** - Run unit tests plus a small RDNA 4 dataset/sample pass that emits trace JSONL, bound artifacts, and AMD-native report. Archive evidence as derived artifacts, not canonical trace changes.

## Anti-Patterns To Avoid

### Mutating Canonical Trace JSONL

Adding `sol_bound`, `hardware_model`, `coverage`, or score fields to `Trace` would break the public benchmark output. Keep all such data in sidecars and score reports.

### Treating Missing Bounds As Zero-Cost

Unsupported operations should produce `unsupported` confidence, warnings, and unscored or guarded scores. A missing or unsupported bound must not become `0.0 ms` in a way that improves score.

### Hard-Coding Release Claims In Formulas

Hardware model validation status belongs in hardware model artifacts, not in operator formulas. Formula code should compute estimates; claim code should read confidence and validation metadata.

### Running Bound Modeling In The Eval Subprocess

`eval_driver.py` is for correctness and timing under isolation. Bound modeling is deterministic post-processing over validated definitions/workloads/traces and should stay outside the user-code subprocess.

## Sources Consulted

- `.planning/PROJECT.md` - v1.9 scope, RDNA 4 validation boundary, CDNA deferrals.
- `.planning/codebase/ARCHITECTURE.md` - current layered architecture and public-contract constraints.
- `src/sol_execbench/core/scoring/amd_sol.py` - current bound artifact, AST extractor, hard-coded hardware model, and estimator baseline.
- `src/sol_execbench/core/scoring/amd_score.py` - current derived AMD-native score integration and warning behavior.
- `scripts/run_dataset.py` - optional derived AMD score report integration point.
- `src/sol_execbench/core/data/trace.py` - canonical trace schema boundary.
- `docs/ARCHITECTURE.md` - user-facing system shape and derived reporting layer.
