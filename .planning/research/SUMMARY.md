# Research Summary: v1.9 AMD SOL/SOLAR Bound Modeling Completion

**Synthesized:** 2026-05-22
**Scope:** RDNA 4-only validation for AMD SOL/SOLAR bound modeling. CDNA 3 /
MI300X real-hardware validation, CDNA 4 validation, and NVFP4/MXFP4 hardware
validation remain deferred.

## Executive Summary

v1.9 should complete the AMD SOL/SOLAR bound-modeling pipeline as a derived
scoring subsystem, not as part of benchmark execution. Canonical trace JSONL,
the primary `sol-execbench` CLI, and public problem schemas should remain
unchanged. New evidence belongs in sidecar AMD SOL bound artifacts, external
hardware model artifacts, and derived AMD-native score reports.

The implementation should not add a new graph or symbolic math dependency.
Use the existing stack: Python 3.12 stdlib `ast`/`json`, dataclasses, Pydantic
v2 where schema validation is useful, pytest, and Ruff. The main architectural
change is splitting the current `amd_sol.py` path into a typed local IR,
extractor, operator estimators, hardware model loader, artifact layer, coverage
summary, and compatibility facade.

## Stack Additions

- No new framework dependency is recommended.
- Keep AST parsing as the first frontend, but do not let raw AST become the
  modeling contract.
- Add focused local scoring modules for:
  - structured bound graph / IR,
  - operator formula and memory accounting,
  - AMD hardware model JSON loading and validation,
  - bound artifact v2 serialization and compatibility,
  - coverage and confidence rollups.
- Use versioned JSON artifacts for hardware models and SOL bounds.
- Keep `amd_score.py` and `scripts/run_dataset.py --amd-score-report` as the
  score/report integration surface.

## Feature Table Stakes

- Structured AMD SOL graph/IR with stable node IDs, op families, tensor roles,
  shape/dtype evidence, dependencies, confidence, and source rationale.
- Explicit operator-family coverage for matmul/bmm, elementwise arithmetic,
  activations, reductions, normalization/RMSNorm/layer norm, softmax/log-softmax,
  data movement/view-like operations, and unsupported calls.
- Auditable FLOP, byte, read/write, intermediate, and memory-movement estimates
  per node.
- Per-node compute bound, memory bound, limiting resource, and aggregate bound.
- Artifact-level confidence and coverage summaries, including worst confidence
  and unsupported/inexact counts.
- Deterministic unsupported/inexact degradation behavior in AMD-native score
  reports.
- External RDNA 4 hardware model artifact with source, architecture, dtype/path,
  clock policy or assumptions, confidence, validation status, and evidence refs.
- Golden tests for parser, formulas, artifact serialization, score integration,
  dataset report integration, and public-contract guardrails.
- Documentation that clearly states RDNA 4-only v1.9 validation and no NVIDIA
  B200/SOLAR/leaderboard equivalence claim.

## Recommended Architecture

The target data flow is:

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

Suggested module split:

| Area | Responsibility |
| --- | --- |
| IR | `BoundGraph`, `BoundNode`, tensor refs, op kind, confidence metadata. |
| Extractor | Convert `Definition.reference` and workload axes into normalized graph nodes. |
| Operators | Formula registry for matmul, elementwise, reductions, normalization, softmax, data movement, unsupported fallback. |
| Estimates | FLOP, byte, memory-movement, and rationale generation. |
| Hardware | Versioned AMD hardware model artifact loading and validation. |
| Artifact | AMD SOL bound artifact v2 build/load/serialize APIs. |
| Coverage | Supported/inexact/unsupported summaries and score eligibility inputs. |
| Facade | Preserve existing imports like `build_amd_sol_bound_artifact()`. |

## Guardrails

- Do not mutate canonical `Trace` JSONL.
- Do not add bound or score fields to `definition.json`, `workload.jsonl`, or
  `solution.json`.
- Do not run bound modeling inside `eval_driver.py`.
- Do not silently drop unsupported operations.
- Do not treat provisional hardware model numbers as hardware-validation claims.
- Do not present `reference_latency` fallback as release-defined optimized
  baseline scoring.
- Do not claim CDNA 3 / MI300X, CDNA 4, NVIDIA B200, upstream SOLAR, or
  leaderboard equivalence in v1.9.

## Key Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Overclaiming AMD scores | Preserve `amd-native-derived` language, warning propagation, docs guardrails, and public-contract tests. |
| Under-modeled memory traffic | Split logical bytes from estimated traffic; record read/write/intermediate/movement rationale. |
| Unsupported ops look complete | Keep unsupported nodes visible and force score warnings or unscored states. |
| Brittle AST parsing | Normalize into IR, add alias/method/chained/multi-output/unsupported fixtures, fail closed. |
| Hardware model folklore | Externalize versioned hardware models with provenance and validation status. |
| Score/baseline misuse | Surface baseline-source counts and require scoring-baseline artifacts for release-style scores. |
| Public schema drift | Keep all new evidence in derived sidecars and score reports. |
| Test blind spots | Add golden positive and negative cases plus RDNA 4 validation closure evidence. |

## Roadmap Implications

Recommended build order:

1. Contract guardrails and hardware model artifact foundation.
2. Structured graph/IR extraction.
3. Operator FLOP/byte/memory-movement estimators.
4. Bound artifact v2 and coverage semantics.
5. AMD score and dataset report integration.
6. Documentation, golden fixtures, and RDNA 4 validation closure.

## Source Research Files

- `.planning/research/STACK.md`
- `.planning/research/FEATURES.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
