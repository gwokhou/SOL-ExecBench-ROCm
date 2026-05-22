# Feature Landscape

**Domain:** AMD SOL/SOLAR bound modeling for SOL ExecBench ROCm
**Milestone:** v1.9 AMD SOL/SOLAR Bound Modeling Completion
**Researched:** 2026-05-22
**Overall confidence:** HIGH for repo-local requirements and current implementation dependencies; MEDIUM for exact operator-priority ordering until real dataset coverage is sampled.

## Scope Decision

v1.9 should complete the modeling pipeline, not broaden the validation claim surface. The product behavior should be: every AMD-native score is backed by a derived, auditable AMD SOL bound artifact; every artifact exposes graph, work, bound, hardware-model, confidence, and coverage evidence; unsupported or inexact analysis remains visible and guarded; RDNA 4 is the only validation target for milestone completion.

The current implementation already has `GraphNode`, `WorkEstimate`, `AmdHardwareModel`, `OpSolBound`, `AmdSolBoundArtifact`, coverage summaries, and AMD-native score reports in `src/sol_execbench/core/scoring/amd_sol.py` and `src/sol_execbench/core/scoring/amd_score.py`. v1.9 should evolve these from a conservative AST visitor into a stable modeling subsystem with explicit IR semantics, external hardware artifacts, golden fixtures, and report integration.

## Table Stakes

Features users and roadmap consumers should expect. Missing any of these leaves the bound-modeling milestone incomplete.

| Feature | Why Expected | Complexity | Dependencies / Notes |
|---------|--------------|------------|----------------------|
| Structured AMD SOL graph/IR schema | Bound artifacts need stable, inspectable structure beyond raw AST-derived expressions. | High | Extend `GraphNode` into a normalized IR with node IDs, op family, tensor roles, shape/dtype evidence, dependency edges, confidence, and rationale. Keep artifact derived and separate from canonical trace JSONL. |
| Explicit operator-family coverage table | Maintainers need to know which SOL ExecBench operations are modeled directly, conservatively, or not at all. | Medium | Build on `_CALL_ANALYZERS` coverage for matmul, reductions, normalization, softmax, activations, and data movement. Add tests asserting coverage summaries and labels. |
| Matmul/GEMM exact-ish modeling | Matrix multiply is a core benchmark family and current `SUPPORTED` path. | Medium | Preserve `2 * output_elements * reduction_dim` evidence, but make K-dimension inference robust for batched matmul and named axis shapes. |
| Elementwise and activation modeling | Common fused and unfused kernels need at least conservative FLOP/byte bounds. | Medium | Keep inexact labels unless formulas are specific. Cover add/sub/mul/div/pow, relu, gelu, silu, sigmoid, tanh, exp, sqrt, rsqrt with per-output-element formulas and dtype-aware bytes. |
| Reduction and normalization modeling | Softmax, RMSNorm, layer norm, reductions are common in model workloads. | High | Model max/sum/mean/var/std/norm/rms_norm/layer_norm/group_norm with pass counts, reduction axes, output shape, read/write bytes, and `INEXACT` confidence unless exact formulas are encoded. |
| Softmax/log-softmax modeling | Softmax-like operations are prominent and already documented as conservative. | Medium | Keep multi-pass estimate evidence: max, exp, sum, normalization, writeback. Label conservative when fusion or implementation details are unknown. |
| Data movement evidence | SOL bounds must not pretend views/transposes/reshapes are compute, but memory movement can dominate real kernels. | High | Distinguish zero-copy views from materializing movement when evidence allows. Track bytes read, bytes written, and "logical view only" vs "materialized/contiguous" confidence. |
| FLOP, byte, and memory-movement breakdown | Roadmap explicitly needs auditable FLOP/byte/memory movement evidence. | High | Split current `bytes_accessed` into read bytes, write bytes, optional intermediate bytes, total bytes, and memory movement rationale. Preserve aggregate compatibility or migrate with schema versioning. |
| Per-node bound calculation | Users need to see compute bound, memory bound, limiting resource, and aggregate bound. | Medium | Current `OpSolBound` already does this. v1.9 should verify no unsupported zero-bound node can silently make an optimistic aggregate. |
| Artifact-level confidence and coverage | Score reports need a single summary that reflects unsupported/inexact content. | Medium | Extend current `coverage_summary` with worst confidence, scored eligibility, counts by op family, and unsupported expressions. |
| Unsupported/inexact degradation behavior | Quality gate requires graceful degradation, not invented precision. | Medium | Unsupported nodes stay in graph and work estimates; scores become guarded or unscored based on policy. Inexact nodes may score only with warnings and claim-level downgrade. |
| External hardware model artifacts | Project goal explicitly says hardware models should not rely only on hard-coded provisional defaults. | High | Add versioned JSON artifacts for `gfx1200` RDNA 4 model inputs with source, dtype/path, bandwidth, peak values, validation status, clock policy, and provenance. Keep `gfx94*` entries unvalidated/deferred. |
| Hardware model loader and validation | External artifacts need schema checks and clear failure modes. | Medium | Add Pydantic/dataclass parser, reject missing architecture, non-positive peak/bandwidth, unknown validation status, and mismatched requested architecture. |
| RDNA 4 validation metadata | v1.9 completion is RDNA 4 only. | Medium | Allow `gfx1200` model entries to be marked validated only when evidence files and tests exist. Do not promote CDNA 3 / MI300X or CDNA 4. |
| AMD-native score integration | Bound modeling must feed the existing derived AMD score report path. | Medium | Continue using `score_amd_native_workload()` / trace workflow. Score output must include bound, hardware model, timing, baseline, confidence, and warnings. |
| Canonical trace immutability | Existing public contract must not change. | Low | Existing tests assert AMD SOL artifacts and scores do not mutate `Trace`. Keep all new fields in derived artifacts/reports. |
| Dataset report integration | Maintainers need suite-level reporting for benchmark batches. | Medium | `scripts/run_dataset.py` already creates AMD score reports. v1.9 should include SOL artifact references and coverage summaries per workload. |
| Golden bound fixtures | Modeling changes must be regression-testable. | Medium | Add deterministic fixtures for matmul, batched matmul, elementwise chain, reduction, softmax, normalization, transpose/reshape, and unsupported op. Assert exact graph, FLOP, byte, bound, confidence, and warnings. |
| Documentation and claim guardrails | Users must understand what AMD SOL/SOLAR means in this ROCm fork. | Medium | Update `docs/analysis.md` and parity/score docs with artifact schema, confidence labels, hardware model provenance, RDNA 4-only validation, and no NVIDIA B200/SOLAR/leaderboard equivalence. |

## Differentiators

Features that make the capability more credible than a simple roofline calculator. These are valuable, but should follow the table-stakes foundation.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Operator IR with dependency edges | Makes fused expressions and multi-stage references auditable instead of just a flat call list. | High | Useful for later fusion-aware bounds and per-node evidence visualization. |
| Shape/dtype provenance per estimate | Lets reviewers trace every FLOP and byte number back to definition/workload axes. | Medium | Store resolved axes, tensor shapes, dtypes, and formula inputs in each estimate or linked evidence block. |
| Memory traffic classification | Distinguishes input read, output write, temporary materialization, view-only movement, and dtype conversion. | High | Important for transpose/contiguous/reshape and normalization/softmax passes. |
| Eligibility policy object | Makes "scored", "guarded scored", and "unscored" deterministic and testable. | Medium | Current score code computes with unsupported warnings. v1.9 can make policy explicit without changing canonical traces. |
| Suite coverage dashboard fields | Helps dataset reports show how much of a run has direct, inexact, or unsupported evidence. | Medium | Add scored/unscored counts, op coverage percentages, worst confidence, and top unsupported op families. |
| Versioned hardware model registry | Allows future RDNA/CDNA model updates without code edits. | Medium | Prefer checked-in JSON under a data/config path over hard-coded defaults. Built-ins can remain as fallback for tests. |
| Golden JSON snapshots | Gives roadmap and review a stable artifact contract. | Low | Use small checked-in expected artifacts; tolerate only deliberate schema-version migrations. |
| Methodology documentation with examples | Makes the feature usable by researchers, not just callable by tests. | Medium | Include one worked example for matmul and one for softmax/normalization showing formulas and confidence labels. |

## Anti-Features

Features to explicitly not build in v1.9.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| CDNA 3 / MI300X validation claims | User explicitly deferred CDNA 3 / MI300X real-hardware validation. | Keep `gfx94*` hardware models unvalidated/provisional and warning-gated. |
| CDNA 4 validation claims | User explicitly deferred CDNA 4 validation. | Do not add validated `gfx12xx`/future CDNA 4 claims without separate evidence. |
| NVIDIA B200, SOLAR, or leaderboard equivalence | This ROCm fork preserves formulas where useful but cannot claim original hardware equivalence. | Use `amd-native-derived` claim language and clear derived-report guardrails. |
| Mutating canonical trace JSONL | Public trace schemas are preserved across prior milestones. | Emit bound and score data as derived artifacts only. |
| Silent unsupported-op omission | Dropping unknown ops makes optimistic SOL bounds look complete. | Keep unsupported nodes visible with zero/unknown work, warnings, and unscored or guarded score behavior. |
| Treating all bytes as exact DRAM traffic | Static modeling cannot know cache behavior, fusion, or implementation-specific movement from reference code alone. | Label estimates as logical bytes or conservative movement evidence with confidence. |
| Fully precise fused-kernel modeling from Python reference AST | Reference code is semantic, not necessarily implementation IR. | Model semantic work and memory evidence; use profiler evidence only as separate timing/correlation input. |
| Full original 124-model extraction pipeline | Explicitly deferred unless only needed as reference context. | Use targeted fixtures and sampled dataset cases for v1.9 validation. |
| New primary CLI behavior or schema surface | The score/bound pipeline is already opt-in and derived. | Integrate through existing dataset and score-report paths. |
| Broad ROCm library performance benchmarking | v1.8 handled library ecosystem support; v1.9 is modeling. | Use library examples only as golden/modeling fixtures if they exercise target op families. |

## Supported Operator Families

Recommended v1.9 coverage target:

| Family | Initial Status | v1.9 Behavior | Confidence Target |
|--------|----------------|---------------|-------------------|
| Matmul / mm / bmm / `@` | Existing supported | Direct FLOP formula, dtype-aware bytes, batched shape support. | `SUPPORTED` for simple dense cases; `INEXACT` for ambiguous/broadcasted cases. |
| Elementwise arithmetic | Existing inexact | One or more ops per output element; chain remains visible as separate or fused IR nodes. | `INEXACT` unless exact expression count is encoded. |
| Activations | Existing inexact | Per-output formulas by activation family where useful; otherwise conservative one-op or multi-op estimate. | `INEXACT`. |
| Reductions | Existing inexact | Axis-aware read/write and operation count for sum/mean/min/max/var/std. | `INEXACT`; `SUPPORTED` only for simple sum/max if formula is exact and tested. |
| Normalization | Existing inexact | Multi-pass estimate for mean/variance or RMS plus scale/apply; expose intermediate/movement rationale. | `INEXACT`. |
| Softmax / log-softmax | Existing inexact | Multi-pass max/exp/sum/div/write formula; axis-aware shape evidence. | `INEXACT`. |
| Data movement / views | Existing inexact | Classify view-only vs materializing operations; bytes are zero/logical/materialized according to evidence. | `INEXACT`; `SUPPORTED` for provably zero-copy metadata-only views if represented. |
| Unsupported complex ops | Existing unsupported | Preserve visible unsupported node and make score/report degradation deterministic. | `UNSUPPORTED`. |

## Feature Dependencies

```text
External hardware model schema -> Hardware model loader -> Bound artifact provenance -> Score report evidence refs
Structured graph/IR -> Operator-family analyzers -> FLOP/byte/movement estimates -> Per-node bounds -> Coverage summary
Confidence labels -> Unsupported/inexact policy -> Score integration warnings/unscored states -> Dataset report summaries
Golden fixtures -> Regression tests -> Documentation examples -> Roadmap acceptance evidence
Canonical trace immutability -> Derived artifacts/reports only -> Public contract guardrail tests
```

## MVP Recommendation

Prioritize:

1. Structured graph/IR and schema-versioned artifact contract.
2. FLOP/byte/memory movement evidence for matmul, elementwise, reductions, normalization, softmax, and data movement.
3. Explicit confidence/degradation policy wired into AMD-native score reports.
4. External `gfx1200` hardware model artifact with RDNA 4 validation metadata and schema tests.
5. Golden fixtures and docs that prove artifact behavior without changing canonical trace JSONL.

Defer:

- CDNA 3 / MI300X and CDNA 4 validation: out of milestone scope.
- Exact cache/fusion modeling: requires implementation/profiler correlation beyond static semantic bounds.
- Full paper dataset extraction: unnecessary for the modeling contract unless later roadmap phases need paper-scale coverage analysis.

## Test Strategy

| Test Area | Required Coverage |
|-----------|-------------------|
| IR extraction | Stable node IDs, op families, expressions, shape/dtype evidence, dependency edges, unsupported node retention. |
| Work estimates | Exact expected FLOP/byte/movement values for small synthetic workloads. |
| Confidence labels | Supported/inexact/unsupported counts and worst-confidence summaries. |
| Hardware models | Valid external `gfx1200` model loads; invalid schema is rejected; `gfx94*` remains unvalidated/deferred. |
| Score integration | Missing/unsupported/inexact bound evidence produces warnings or unscored states; evidence refs survive report serialization. |
| Trace contract | Canonical `Trace.model_dump()` is unchanged before/after bound and score generation. |
| Dataset integration | AMD score report includes per-workload SOL refs and aggregate coverage without requiring full dataset execution. |
| Documentation guardrails | Docs contain RDNA 4-only validation language and no NVIDIA/B200/SOLAR equivalence claims. |

## Sources

- `.planning/PROJECT.md` - v1.9 goals, RDNA 4-only validation scope, CDNA 3/CDNA 4 deferrals.
- `docs/analysis.md` - current AMD-native score and AMD SOL coverage semantics.
- `docs/original_parity.md` - preserved public surfaces and out-of-scope NVIDIA leaderboard equivalence.
- `src/sol_execbench/core/scoring/amd_sol.py` - current graph, work estimate, hardware model, bound artifact, and coverage implementation.
- `src/sol_execbench/core/scoring/amd_score.py` - current derived AMD-native scoring, warnings, and evidence refs.
- `tests/sol_execbench/test_amd_sol_bounds.py` - current bound artifact and confidence coverage tests.
- `tests/sol_execbench/test_amd_native_score.py` - current score integration and guardrail tests.
