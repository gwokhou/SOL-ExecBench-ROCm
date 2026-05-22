# Domain Pitfalls

**Domain:** AMD SOL/SOLAR-like bound modeling for SOL ExecBench ROCm
**Milestone:** v1.9 AMD SOL/SOLAR Bound Modeling Completion
**Researched:** 2026-05-22
**Overall confidence:** HIGH for repo-specific contract and scoring risks; MEDIUM for exact phase names because v1.9 roadmap has not been written yet.

## Critical Pitfalls

Mistakes that can invalidate the milestone or force rewrites.

### Pitfall 1: Overclaiming AMD Scores As SOLAR, B200, Leaderboard, Or Hardware Validation

**What goes wrong:** Derived AMD-native score reports are presented as equivalent to upstream NVIDIA B200/SOLAR or leaderboard results, or as validated AMD hardware-performance claims when the bound model, timing evidence, hardware model, or validation scope is incomplete.

**Why it happens:** The repository preserves the original `sol_score()` formula while adding AMD-native interpretation artifacts. That makes it easy to conflate formula compatibility with claim compatibility.

**Consequences:** Public docs become misleading, downstream users compare incompatible results, and v1.9 violates existing parity/compliance guardrails.

**Warning signs:**
- Docs say "SOLAR parity", "B200 equivalent", "leaderboard", "validated AMD SOL", or "paper-equivalent" without explicit caveats.
- `AmdNativeScore.claim_level` becomes more authoritative than `amd-native-derived`.
- Score reports omit warnings when hardware models are provisional, coverage is unsupported/inexact, or baseline source is `reference_latency`.
- Dataset reports compute suite means across incomplete or unsupported evidence without preserving unscored counts.

**Prevention:**
- Keep AMD-native scores as derived artifacts with `canonical_output: trace_jsonl`.
- Preserve explicit warnings for unsupported operations, unvalidated hardware models, CDNA 3 deferral, provisional reference baselines, and incomplete evidence.
- Require score docs to state "not NVIDIA B200, SOLAR, or leaderboard equivalence claims".
- Add public-contract tests that grep docs and score payloads for no-claim wording and warning propagation.

**Phase placement:** Score Integration and Claim Guardrails phase; final Documentation and RDNA 4 Closure phase must recheck it.

**Confidence:** HIGH. Existing `docs/analysis.md`, `docs/original_parity.md`, `docs/compliance.md`, and `src/sol_execbench/core/scoring/amd_score.py` already encode these constraints.

### Pitfall 2: Under-Modeling Memory Traffic While Reporting Tight Bounds

**What goes wrong:** The bound model counts only input and output tensor bytes, then treats the result as a credible roofline-style memory bound for operations that require intermediate reads/writes, multiple passes, cache-hostile access, reductions, broadcasts, transposes, dtype conversion, temporary materialization, or library workspace.

**Why it happens:** The current `estimate_work()` uses a coarse `_tensor_bytes()` helper and labels many non-matmul operations as `INEXACT`. v1.9 aims to broaden coverage, which increases pressure to turn conservative estimates into apparently exact numbers.

**Consequences:** Memory-bound kernels look artificially close to speed-of-light, model rankings become distorted, and unsupported movement costs are hidden inside a single optimistic byte estimate.

**Warning signs:**
- Data movement nodes have zero FLOPs but no explicit byte rationale.
- Broadcast, view, transpose, reshape, `contiguous`, and dtype-conversion behavior is treated as free or always equivalent.
- Reductions, normalization, and softmax estimate one tensor read/write despite multi-pass behavior.
- The artifact lacks per-node `bytes_accessed` provenance or cannot explain whether bytes are logical tensor bytes, estimated traffic, or measured traffic.

**Prevention:**
- Split logical tensor bytes from estimated memory traffic.
- Record per-node traffic categories: input read, output write, intermediate read/write, reduction pass, materialization, workspace, and unknown traffic.
- Keep inexact memory estimates labeled `INEXACT`; unsupported traffic must remain visible and must degrade score confidence.
- Add golden tests for softmax, RMSNorm/layer norm, reductions, broadcasted elementwise ops, transpose/contiguous, dtype conversion, and fused matmul plus activation.

**Phase placement:** FLOP/Byte/Memory-Movement Modeling phase, after the IR phase and before score integration.

**Confidence:** HIGH. Current `_tensor_bytes()` and coverage semantics make this the central modeling risk.

### Pitfall 3: Treating Recognized But Unsupported Operations As Complete Coverage

**What goes wrong:** An operation appears in the graph and receives a placeholder estimate, but downstream reporting treats it as fully supported.

**Why it happens:** The existing artifact has `SUPPORTED`, `INEXACT`, and `UNSUPPORTED` labels, but the aggregate bound is still a numeric sum. Without strict gates, the presence of a number can be mistaken for complete evidence.

**Consequences:** Coverage summaries lose meaning, score reports become overconfident, and future analyzer work cannot tell real support from placeholder support.

**Warning signs:**
- `unsupported_ops > 0` but a score is displayed without a prominent warning.
- `inexact_ops > 0` is hidden from suite summaries.
- Unknown calls are dropped from the graph instead of emitted as unsupported nodes.
- "coverage" is measured only by graph extraction success, not by estimate confidence.

**Prevention:**
- Define score states separately from numeric availability: `scored_complete`, `scored_inexact`, `unscored_unsupported`, and `unscored_missing_evidence`.
- Keep unsupported nodes visible in `graph_nodes`, `work_estimates`, `coverage_summary`, and suite rollups.
- Require coverage tests where unknown calls survive extraction and force `UNSUPPORTED_EVIDENCE_WARNING`.
- Treat `INEXACT` as usable only for guarded derived reports, not for hardware-validation claims.

**Phase placement:** Structured IR and Coverage Semantics phase owns labels; Score Integration phase owns degradation behavior.

**Confidence:** HIGH. `amd_sol.py` and `amd_score.py` already contain the confidence vocabulary and warning path.

### Pitfall 4: Brittle AST Parsing Mistaken For A Structured Operator Model

**What goes wrong:** The v1.9 graph remains a syntax visitor over Python source but is described as a robust workload IR. Semantically equivalent code forms produce different bounds or silently missed operations.

**Why it happens:** Current extraction uses `ast.parse(definition.reference)` plus call-name and binary-operator heuristics. This works for simple references but is fragile for aliases, imports, helper functions, chained expressions, loops, conditionals, `torch.nn.functional`, Tensor methods, `einsum`, indexing, slicing, and multiple outputs.

**Consequences:** Bound artifacts are unstable across reference-code style, tests overfit to simple examples, and downstream roadmap work inherits a parser that cannot support paper-style SOL/SOLAR modeling.

**Warning signs:**
- Tests only cover `a @ b` and direct `torch.softmax(x)` forms.
- Aliased imports such as `F.softmax`, helper wrappers, or method chains are unsupported with no explicit rationale.
- Loop bodies, conditionals, indexing, slicing, tuple outputs, and shape-dependent branches are ignored.
- `ast.unparse()` expression strings become the primary identity instead of stable node IDs and normalized op attributes.

**Prevention:**
- Introduce a normalized internal graph/IR with explicit op type, inputs, outputs, shapes, dtype, confidence, source span or source expression, and analyzer provenance.
- Keep AST extraction as one frontend, not the model itself.
- Add fixture references for aliases, Tensor methods, helper functions, chained calls, multi-output returns, indexing/slicing, and unsupported dynamic control flow.
- Fail closed: unsupported syntax must create unsupported evidence or a clear extraction diagnostic, not disappear.

**Phase placement:** First phase: Structured Graph/IR and Analyzer Frontend. Later modeling phases should consume only the IR.

**Confidence:** HIGH. The current `_GraphVisitor` is intentionally conservative and should not become the long-term contract.

### Pitfall 5: Hardware Model Provenance Becomes Hard-Coded Folklore

**What goes wrong:** Peak TFLOPs and bandwidth constants remain embedded in code or are updated without source, clock policy, dtype path, architecture scope, or validation status.

**Why it happens:** `default_amd_hardware_models()` currently provides provisional built-ins for `gfx1200` and unvalidated `gfx942`. v1.9 explicitly wants external hardware model artifacts, but it is easy to externalize only the numbers without externalizing provenance.

**Consequences:** Reports are not auditable, RDNA 4 validation claims cannot be reproduced, and deferred CDNA 3/CDNA 4 scopes may accidentally inherit RDNA 4 evidence.

**Warning signs:**
- Hardware model files lack source URLs/doc references, measurement method, clock policy, ROCm version, dtype path, or validation status.
- `gfx94*` models appear as validated during v1.9.
- Hardware model refs are missing from score reports.
- Tests assert only numeric bounds, not provenance fields.

**Prevention:**
- Store hardware models as versioned derived inputs with schema version, architecture, dtype/path, peak compute, bandwidth, clock policy, source, confidence, validation status, and validation evidence references.
- Require `hardware_model_ref` in score reports for release-style reports.
- Keep `gfx1200` as the only v1.9 validation target; keep CDNA 3/CDNA 4 entries unvalidated or out of scope.
- Add schema and golden tests that fail when provenance or validation fields are missing.

**Phase placement:** Hardware Model Artifact and Provenance phase, before score integration and before RDNA 4 closure.

**Confidence:** HIGH. Existing tests already check hardware model evidence survives bound and score artifacts.

### Pitfall 6: Misusing Baselines And Scores As Absolute Performance Measures

**What goes wrong:** AMD-native score reports mix measured latency, PyTorch reference latency, optimized scoring baselines, and SOL bounds without clearly distinguishing their roles.

**Why it happens:** The code permits fallback from scoring-baseline artifacts to `trace.evaluation.performance.reference_latency_ms`, with a warning. That is useful for development but dangerous for release claims.

**Consequences:** A provisional reference baseline can be interpreted as a release-defined optimized baseline, and score changes may reflect baseline drift rather than kernel performance or bound quality.

**Warning signs:**
- Release docs show scores with `baseline_source: reference_latency`.
- Suite mean includes scores from mixed baseline sources without a visible summary.
- Missing or failed traces still contribute to aggregate statistics.
- Score comparisons are made across machines without architecture, ROCm version, clock policy, and timing evidence alignment.

**Prevention:**
- Require optimized scoring baseline artifacts for release-defined v1.9 scores.
- Surface baseline-source counts in suite reports and docs.
- Exclude incomplete and failed traces from numeric means while preserving `unscored_count`.
- Add tests for missing baseline, reference fallback warning, mixed baseline summaries, and failed trace behavior.

**Phase placement:** Score Integration and Dataset Reporting phase.

**Confidence:** HIGH. `amd_score.py` already distinguishes `scoring_baseline`, `reference_latency`, and `missing`.

### Pitfall 7: Public Schema Drift To Accommodate Bound Artifacts

**What goes wrong:** Bound or score fields are added to canonical trace JSONL, primary `sol-execbench` CLI defaults, public definition/workload/solution schemas, or eval-driver contracts.

**Why it happens:** Bound modeling needs additional evidence. The easiest implementation path is often to attach it to existing traces, but this repository has repeatedly chosen derived sidecar artifacts to preserve SOL ExecBench public contracts.

**Consequences:** Existing users and dataset tools break, original parity weakens, and public-contract guardrails regress.

**Warning signs:**
- New trace fields such as `amd_sol`, `amd_score`, `roofline`, or `coverage_summary`.
- New primary CLI options for derived workflows appear in `sol-execbench --help`.
- Pydantic public field names or enum semantics change for non-modeling reasons.
- Eval driver starts performing bound analysis during benchmark execution.

**Prevention:**
- Keep bound artifacts and score reports as opt-in derived outputs.
- Do not mutate canonical trace JSONL; only reference traces by path or workload UUID.
- Extend existing public-contract tests to block new primary CLI options and trace fields.
- Put modeling CLIs, if any, in scripts or secondary commands that consume existing artifacts.

**Phase placement:** Cross-cutting guardrail from phase 1; explicit Contract Guardrails and Documentation phase before closure.

**Confidence:** HIGH. Existing public-contract tests already enforce noncanonical derived artifacts and primary CLI stability.

### Pitfall 8: Test Blind Spots Around Coverage, Negative Cases, And RDNA 4 Validation

**What goes wrong:** Tests prove the happy path for one matmul and one score, but not the failure modes that protect users from bad claims.

**Why it happens:** Bound modeling has many small branches: parser coverage, confidence propagation, byte modeling, hardware provenance, score warnings, baseline fallback, schema stability, and docs wording. Unit tests can pass while the model is still unsafe.

**Consequences:** Regressions appear in documentation or reporting rather than in tests, and v1.9 closure lacks evidence for the risks it claims to address.

**Warning signs:**
- No golden artifacts are checked into tests for representative operator families.
- No tests assert unsupported operations remain visible.
- No tests verify docs claim scope for RDNA 4-only validation.
- Real RDNA 4 validation command output is not recorded in milestone closure artifacts.

**Prevention:**
- Add golden bound artifact tests for matmul, batched matmul, elementwise, activation, reduction, softmax, normalization, data movement, mixed/fused patterns, and unsupported calls.
- Add negative tests for incomplete evidence, unvalidated hardware, unsupported ops, missing baselines, failed traces, and CDNA 3/CDNA 4 no-claim wording.
- Add public-contract guardrails for canonical trace stability and primary CLI stability.
- Record RDNA 4-only validation evidence in closure docs, including exact commands and environment assumptions.

**Phase placement:** Dedicated Golden Tests and RDNA 4 Validation phase, with test additions required in every earlier phase.

**Confidence:** HIGH. Existing tests cover some guardrails, but v1.9 expands the modeling surface substantially.

## Moderate Pitfalls

### Pitfall 1: Aggregating Per-Op Bounds Too Naively

**What goes wrong:** The artifact sums each per-op max(compute, memory) bound and calls it an aggregate SOL bound, even for fused kernels, overlapped operations, or library calls where execution does not correspond to independent graph nodes.

**Prevention:** Label aggregation method explicitly, add fusion-aware rationale where known, and keep aggregate bounds inexact unless the graph-to-kernel mapping is validated.

**Phase placement:** Bound Aggregation and Score Integration phase.

### Pitfall 2: Shape And Axis Resolution Errors

**What goes wrong:** FLOP and byte formulas use unresolved or wrong axes, especially for reductions, batched matmul, broadcasting, symbolic dimensions, and multiple outputs.

**Prevention:** Normalize resolved shapes in the IR, include shapes in golden artifacts, and test symbolic axes with workload-specific values.

**Phase placement:** Structured IR phase and FLOP/Byte Modeling phase.

### Pitfall 3: Dtype Path Ambiguity

**What goes wrong:** A bound uses FP32, BF16, FP16, FP8, or packed FP4 assumptions without tying them to problem dtype, accumulation dtype, or hardware path.

**Prevention:** Carry input/output dtype, accumulation dtype when known, and hardware-model `dtype_or_path` through every artifact; unsupported dtype paths must degrade confidence.

**Phase placement:** Hardware Model Artifact phase and FLOP/Byte Modeling phase.

### Pitfall 4: Timing Evidence Misalignment

**What goes wrong:** Score reports combine bounds for one architecture or clock policy with timing evidence from another run.

**Prevention:** Require evidence refs and metadata checks for architecture, ROCm version when available, clock policy, timing backend, warmups, measured iterations, and source-specific fallback reason.

**Phase placement:** Score Integration and RDNA 4 Validation phase.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
| --- | --- | --- |
| Structured Graph/IR and Analyzer Frontend | Brittle AST parser becomes the public model | Build a normalized IR; keep AST as frontend; unsupported syntax emits visible unsupported evidence |
| Operator Coverage Semantics | Recognized operations are treated as complete | Preserve `SUPPORTED`/`INEXACT`/`UNSUPPORTED`; add coverage rollups and negative tests |
| FLOP, Byte, and Memory-Movement Modeling | Memory traffic is under-modeled but reported as exact | Split logical bytes from estimated traffic; add per-node rationale and golden memory cases |
| Hardware Model Artifacts | Constants lack provenance or validation scope | Externalize versioned hardware model files with source, clock policy, confidence, and validation status |
| AMD Score Integration | Scores hide incomplete evidence, reference fallback, or unsupported ops | Keep guarded score states, warnings, baseline-source summaries, and unscored counts |
| Public Contract and Docs | Bound artifacts mutate canonical trace or imply B200/SOLAR parity | Keep sidecar derived artifacts; test CLI/schema/trace stability and no-claim wording |
| RDNA 4 Validation Closure | Validation scope creeps to CDNA 3/CDNA 4 or lacks evidence | Record RDNA 4 commands/evidence; keep `gfx94*` and CDNA 4 explicitly deferred |

## Actionable Guardrails For Requirements

- Every bound artifact must include `schema_version`, `derived: true`, workload identity, graph nodes, work estimates, op bounds, aggregate bound, coverage summary, hardware model provenance, and confidence labels.
- Unsupported and inexact operations must be visible in artifacts and must propagate to score warnings or unscored states.
- Release-style AMD scores must require scoring-baseline artifacts; `reference_latency` fallback remains development-only and warning-labeled.
- Canonical trace JSONL and primary `sol-execbench` CLI must remain unchanged.
- Hardware model artifacts must include source, architecture, dtype/path, clock policy or measurement assumptions, validation status, and evidence references.
- v1.9 validation claims must say RDNA 4 only; CDNA 3 / MI300X and CDNA 4 remain deferred.
- Golden tests must cover both positive operator families and negative evidence-degradation cases.

## Sources

- `.planning/PROJECT.md` - v1.9 milestone goal, scope, RDNA 4 validation boundary, and deferred CDNA 3/CDNA 4 work. Confidence: HIGH.
- `docs/analysis.md` - AMD-native score interpretation, coverage semantics, derived artifact constraints, and timing evidence guidance. Confidence: HIGH.
- `docs/original_parity.md` - original SOL ExecBench public surface and no NVIDIA leaderboard equivalence. Confidence: HIGH.
- `docs/compliance.md` - unsupported NVIDIA runtime features, known gaps, and hardware-validation caveats. Confidence: HIGH.
- `src/sol_execbench/core/scoring/amd_sol.py` - current AST extraction, FLOP/byte estimates, hardware model metadata, coverage summary, and bound aggregation behavior. Confidence: HIGH.
- `src/sol_execbench/core/scoring/amd_score.py` - derived AMD score reports, warnings, baseline-source handling, and score support conditions. Confidence: HIGH.
- `tests/sol_execbench/test_public_contract_guardrails.py` - existing tests for public contract stability, derived artifacts, hardware model evidence, and claim guardrails. Confidence: HIGH.
