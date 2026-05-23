# Domain Pitfalls

**Domain:** Paper-aligned SOLAR automatic derivation for SOL ExecBench ROCm
**Milestone:** v1.10 paper-aligned SOLAR automatic derivation
**Researched:** 2026-05-23
**Overall confidence:** HIGH for local contract risks; MEDIUM for paper-alignment interpretation because this research uses the arXiv 2603.19173 abstract baseline rather than the full implementation.

## Baseline

The paper baseline is arXiv 2603.19173, submitted March 19, 2026. Its abstract frames SOL-ExecBench as a CUDA/NVIDIA Blackwell benchmark with 235 problems extracted from 124 models, analytically derived Speed-of-Light bounds computed by SOLAR, a SOL Score measuring how much gap a candidate closes between a release-defined scoring baseline and the hardware SOL bound, and robustness support through sandboxing, GPU clock locking, L2 cache clearing, isolated subprocess execution, and static-analysis checks.

This ROCm milestone must not recreate the paper's dataset extraction, hosted leaderboard, NVIDIA Blackwell target, or new real-hardware validation. The safe target is narrower: improve automatic derivation correctness for AMD SOL v2 sidecars, BoundGraph IR, operator estimates, coverage semantics, and AMD-native derived score reports while preserving the existing public contracts.

## Critical Pitfalls

### Pitfall 1: Claiming Paper SOLAR Parity From A Narrow ROCm Derivation Milestone

**What goes wrong:** Documentation, score reports, CLI help, or release notes imply "paper-equivalent SOLAR", "full SOLAR", "leaderboard-ready", "NVIDIA B200 equivalent", or "hardware-validated" results when v1.10 only improves derivation inside a ROCm fork.

**Why it happens:** The milestone name says paper-aligned SOLAR, while the paper abstract includes a much larger benchmark story: 124-model extraction, 235 problems, NVIDIA Blackwell hardware targets, fixed SOL targets, and a robust evaluation harness. Those broader claims are explicitly out of scope here.

**Consequences:** Users can misinterpret AMD-native derived scores as paper results, compare them against NVIDIA leaderboard claims, or trust provisional bounds as measured hardware validation.

**Warning signs:**
- Docs use "SOLAR parity", "full SOLAR", "paper-complete", "leaderboard", "B200", or "validated" without an immediate scope qualifier.
- `claim_level` is renamed away from `amd-native-derived`.
- Score warnings are removed because a workload now has a richer derived graph.
- Tests stop checking CDNA 3 / MI300X validation deferral and no NVIDIA/SOLAR/leaderboard equivalence wording.

**Prevention strategy:**
- Keep "paper-aligned derivation" as the strongest claim, not "paper-equivalent benchmark".
- Preserve existing AMD-native score warnings for degraded, unscored, provisional hardware-model, reference-baseline, and CDNA 3 / MI300X cases.
- Add v1.10 guardrail tests that scan docs and public help for out-of-scope claims.
- Require every new public-facing artifact to distinguish derived evidence from canonical trace JSONL and hardware validation.

**Detection:** A release-note or docs diff can be read without seeing "derived", "ROCm", "AMD-native", and "not leaderboard / B200 / hardware validation" near SOLAR wording.

**Phase to address:** Phase 1: Scope, Claim Contract, And Acceptance Gates. Recheck in the final Documentation and Guardrail Closure phase.

### Pitfall 2: Inflating Scores With Partial Coverage

**What goes wrong:** A workload with unsupported or inexact operations still receives a normal-looking SOL Score because supported subgraphs are summed and missing work contributes zero bound time.

**Why it happens:** Current `amd_sol_v2` sidecars score `degraded` artifacts when evidence is inexact, but mark artifacts `unscored` only when any operation is `UNSUPPORTED`. New v1.10 families such as attention, MoE, convolution, SSM/Mamba, embedding/positional, and linear projection will create many opportunities for partial extraction. If unmatched branches, fused subexpressions, masks, routing, or dynamic shape regions are simply absent from the graph, coverage looks better than it is.

**Consequences:** Scores can improve by omission: a candidate appears closer to the SOL bound because the bound is missing expensive work. Suite means become biased toward workloads with easier extraction.

**Warning signs:**
- `coverage_summary.total_ops` is low for a complex reference, but no warning explains why.
- `aggregate_bound.status == "scored"` even though extraction had fallback, skipped nodes, dynamic branches, or missing tensors.
- New unsupported families return zero-work `INEXACT` estimates instead of `UNSUPPORTED`.
- Suite reports emphasize `mean_score` without `scored_count`, `unscored_count`, and coverage warnings.

**Prevention strategy:**
- Treat missing graph regions as explicit unsupported evidence, not absence.
- Add a coverage denominator that records extracted, skipped, unsupported, and unmodeled source operations where possible.
- Require operation-family coverage thresholds before an artifact can be `scored`; below threshold must be `degraded` or `unscored`.
- Keep unsupported operation presence as an `unscored` gate unless a documented conservative upper-bound strategy exists for that family.
- Add golden tests where intentionally partial attention/MoE/conv extraction cannot produce a normal score.

**Detection:** A complex reference produces fewer nodes than expected, no `graph_warning:*` or `estimate_warning:*` appears, and `score` is non-null.

**Phase to address:** Phase 2: Extraction Coverage Semantics; Phase 5: Score Eligibility And Suite Reporting.

### Pitfall 3: Shape And DType Errors Corrupt Bounds Silently

**What goes wrong:** FLOP and byte formulas use the wrong axes, layout, broadcast semantics, batch dimensions, packed dtype widths, or output metadata.

**Why it happens:** Existing estimates rely heavily on `BoundTensor.shape`, dtype byte tables, `torch.fx` shape propagation, and fallback heuristics such as first-input shape or output shape. v1.10 operation families are more shape-sensitive: attention has heads, sequence axes, causal masks, KV-cache variants, and softmax axes; MoE has top-k routing and sparse expert batches; convolution has stride/padding/dilation/groups/layout; SSM/Mamba has scan/state dimensions; embeddings use index dtype and gathered output shape.

**Consequences:** Bounds can be wrong by orders of magnitude while still looking precise because formulas and inputs are present.

**Warning signs:**
- Formula inputs are derived from tensor rank alone, not named semantic axes.
- Dtype conversion maps paper-only FP8/NVFP4-style concepts to AMD byte widths without explicit support status.
- Broadcast views or `expand` are counted as materialized movement, or `contiguous`/copy-like operations are counted as zero movement.
- Convolution formulas ignore groups, dilation, padding, or output spatial shape.
- Attention formulas ignore mask type, softmax axis, dropout/training state, KV-cache length, or multi-query/grouped-query layout.

**Prevention strategy:**
- Introduce family-specific semantic attributes before formulas: e.g. `batch`, `seq_q`, `seq_k`, `heads_q`, `heads_kv`, `head_dim`, `top_k`, `experts`, `groups`, `kernel_shape`, `stride`, `padding`, `state_dim`.
- Make unknown semantic axes `INEXACT` or `UNSUPPORTED`; never infer them only from common rank patterns when multiple interpretations exist.
- Keep byte formulas tied to explicit source/target dtypes and tensor roles; packed/low-precision formats need separate evidence and warnings.
- Add formula-input golden tests for each new family across non-square, batched, broadcasted, and dtype-varied cases.

**Detection:** Two workloads with different semantic axes but similar ranks produce identical formula inputs; dtype warnings disappear for low-precision paths without new validation evidence.

**Phase to address:** Phase 3: Family-Specific Semantic Extraction; Phase 4: Operator Formula And Byte Modeling.

### Pitfall 4: FX/AST Fragility Creates Non-Deterministic Derivation

**What goes wrong:** The same mathematical reference gets different BoundGraph nodes depending on code style, PyTorch version behavior, module wrapping, Python syntax, or whether FX tracing succeeds.

**Why it happens:** Current graph extraction tries `torch.fx.symbolic_trace` plus shape propagation, then falls back to AST parsing. FX can fail on dynamic control flow, data-dependent shapes, Python-side helper functions, modules, and unsupported calls. AST fallback can classify call names but cannot reliably recover runtime tensor metadata for complex expressions.

**Consequences:** Bounds are unstable and hard to audit. A harmless reference rewrite can change score eligibility or operation-family classification.

**Warning signs:**
- Tests only cover one spelling of each operation.
- New family extraction is implemented only as string matching on source expressions.
- FX failure is recorded only as `dynamic_trace_failed` while AST fallback still marks rich operations as supported.
- Helper functions, module methods, `torch.nn.functional`, method calls, and operator overloads classify differently.

**Prevention strategy:**
- Define a canonical extractor pipeline with explicit source provenance per node: `torch.fx`, AST, manual pattern, or unsupported fallback.
- Keep AST-only family matches conservative unless shape/dtype semantics are independently available.
- Add equivalence tests for common spellings: function call, method call, module call, operator overload, helper wrapper, and nested expression.
- Record extraction instability as warnings and degrade score eligibility when frontends disagree.
- Prefer structured PyTorch/FX metadata over source-string parsing whenever available.

**Detection:** Rewriting `torch.matmul(a, b)` to `a @ b`, wrapping `F.linear`, or using an `nn.Module` changes aggregate status or removes warnings.

**Phase to address:** Phase 2: Extractor Frontend Robustness; Phase 3: Family-Specific Semantic Extraction.

### Pitfall 5: Unsupported Operations Become Taxonomy-Only Placeholders

**What goes wrong:** New operation families are added to `OpFamily` and coverage summaries, but their estimates remain generic zero-work or pointwise placeholders.

**Why it happens:** v1.9 already has operation-family names beyond fully modeled estimates. v1.10 explicitly targets attention, MoE, convolution, SSM/Mamba, embedding/positional, and linear projection paths. It is tempting to mark a recognized pattern as covered before deriving auditable formula inputs.

**Consequences:** Coverage increases on paper while bound correctness does not. Unsupported-heavy workloads can appear as degraded rather than unscored.

**Warning signs:**
- New family nodes have `formula_kind="unsupported"` or generic `output_elements` but `confidence != UNSUPPORTED`.
- `rationale` says "recognized" but does not explain formula source, semantic axes, or missing assumptions.
- `op_family_counts` grows while `operator_work_estimates` lack family-specific formulas.
- Tests assert only classification, not formula inputs and aggregate status.

**Prevention strategy:**
- Require a family promotion checklist: extraction, semantic attributes, formula, byte buckets, confidence rules, warning rules, round-trip sidecar parsing, and score eligibility tests.
- Keep classification-only support as `UNSUPPORTED` or `INEXACT` with explicit "taxonomy only" warnings.
- For each new family, add negative tests where missing required attributes force `UNSUPPORTED`.

**Detection:** A new family can be counted in coverage without at least one golden estimate that checks FLOPs, read bytes, write bytes, intermediate bytes, movement bytes, confidence, and rationale.

**Phase to address:** Phase 4: Family Model Promotion Gates.

### Pitfall 6: Aggregation Assumes Serial Sum For Fused Or Overlapped Work

**What goes wrong:** Aggregate SOL bound time is computed by summing per-op max(compute, memory) bounds even when the extracted workload represents fused kernels, overlapped movement, shared intermediates, or operations that should be considered as a compound pattern.

**Why it happens:** Current v2 aggregation sums `op_bounds`. That is a reasonable conservative starting point for local modeling, but paper-aligned SOLAR derivation may require compound bounds for attention, fused MLPs, normalization plus activation, or MoE dispatch/combine.

**Consequences:** Bound estimates may be pessimistic or internally inconsistent. SOL Score can be distorted because `t_sol` no longer represents a coherent hardware speed-of-light target.

**Warning signs:**
- Intermediate bytes are counted as global memory traffic even when the intended fused pattern keeps values in registers/LDS.
- Attention is modeled as independent GEMM, softmax, GEMM with no compound rationale.
- MoE dispatch and combine are counted as ordinary data movement without routing sparsity semantics.
- Aggregate rationale does not state whether the bound is serial, fused, or conservative.

**Prevention strategy:**
- Add `aggregate_strategy` or equivalent evidence for serial, compound, or conservative aggregation.
- For complex families, model both decomposed node evidence and family-level compound bound rationale.
- Keep inexact confidence when fusion/overlap assumptions are unknown.
- Do not use compound aggregation to remove unsupported child evidence; unsupported subcomponents still gate score eligibility.

**Detection:** A fused reference and a decomposed reference get the same aggregate strategy with no warning.

**Phase to address:** Phase 5: Aggregate Bound And Score Eligibility.

### Pitfall 7: Public Contract Drift From Adding Derivation Outputs

**What goes wrong:** Bound graph, operator estimates, coverage summaries, hardware model refs, or SOLAR flags leak into canonical `definition.json`, `workload.jsonl`, trace JSONL, or primary `sol-execbench` CLI defaults.

**Why it happens:** Derivation data is useful and developers may put it near existing benchmark data rather than keeping it in sidecars and opt-in reports. Existing guardrails already protect public schemas and primary CLI help.

**Consequences:** Existing users and tests break. The ROCm fork diverges from SOL ExecBench public semantics beyond the intended derived-report layer.

**Warning signs:**
- New fields such as `bound_graph`, `operator_work_estimates`, `coverage_summary`, `aggregate_bound`, or `hardware_model_ref` appear in canonical models.
- Primary CLI help exposes `--sol-bound-v2`, `--bound-graph`, or similar derived workflow options.
- Trace JSONL changes to include score or bound evidence.

**Prevention strategy:**
- Keep all v1.10 derivation outputs as sidecars or noncanonical reports with `derived=True` and `canonical_output="trace_jsonl"` where applicable.
- Extend `test_public_contract_guardrails.py` for any new derived artifact names.
- Preserve primary CLI behavior; use existing dataset/reporting opt-in surfaces for derived workflows.

**Detection:** Existing public contract guardrail tests need updates because they fail after a derivation change. Treat that as a blocker unless the milestone explicitly approves a public contract change.

**Phase to address:** Phase 1: Contract Baseline; Phase 6: Documentation, Guardrails, And Closure.

## Moderate Pitfalls

### Pitfall 8: Release-Defined Baseline Semantics Get Mixed With SOL Bound Semantics

**What goes wrong:** Score logic treats a better-derived SOL bound as if it also improves or validates the scoring baseline.

**Prevention strategy:** Keep `baseline_source` independent from SOL bound derivation. Continue warning for `reference_latency` fallback and require scoring baseline artifacts for release-defined scoring. Add tests proving v1.10 bound changes do not alter baseline selection.

**Phase to address:** Phase 5: Score Eligibility And Suite Reporting.

### Pitfall 9: Robustness Claims Escape The Milestone Boundary

**What goes wrong:** Because the paper abstract mentions sandboxing, clock locking, L2 cache clearing, isolated subprocess execution, and static-analysis checks, v1.10 docs imply new robustness validation even though the milestone focuses on derivation.

**Prevention strategy:** Reference existing harness/guardrail work only as inherited project context. Do not claim new robustness coverage unless tests are actually added. Any robustness wording should say v1.10 preserves existing guardrails while improving derived evidence.

**Phase to address:** Phase 6: Documentation, Guardrails, And Closure.

### Pitfall 10: Hardware Model Provenance Becomes Implicit

**What goes wrong:** New derivation artifacts omit hardware model source, validation status, architecture, or model confidence because the formulas are the focus.

**Prevention strategy:** Preserve strict hardware model refs in every sidecar and score evidence ref. v1.10 should not upgrade `hardware_validation_status` or `model_validation_status` without real hardware validation evidence, which is out of scope.

**Phase to address:** Phase 5: Aggregate Bound And Score Eligibility; Phase 6: Closure.

### Pitfall 11: Tests Overfit To Toy Operators

**What goes wrong:** Golden tests pass for `a @ b`, `softmax`, and simple reductions but miss realistic reference structure: nested modules, multiple outputs, in-place-looking patterns, dynamic axes, masks, mixed dtypes, and fused expressions.

**Prevention strategy:** Build a compact fixture matrix per target family with at least one positive, one inexact, one unsupported, one alternate spelling, and one non-square/nontrivial-shape case. Do not require real GPU validation for these derivation tests.

**Phase to address:** Every implementation phase; Phase 6 owns final matrix audit.

## Minor Pitfalls

### Pitfall 12: Warning Strings Become Unstable

**What goes wrong:** Tests and downstream tools cannot rely on warning categories because new extractor/model paths emit ad hoc messages.

**Prevention strategy:** Keep machine-stable prefixes such as `graph_warning:`, `estimate_warning:`, `unsupported_operator:`, `inexact_operator:`, `aggregate_degraded:`, and `aggregate_unscored:`. Add new categories only through documented constants or tests.

**Phase to address:** Phase 2: Coverage Semantics.

### Pitfall 13: Rationale Text Is Present But Not Auditable

**What goes wrong:** Estimates include generic rationales that do not identify assumptions, formula inputs, or confidence reasons.

**Prevention strategy:** Require each new family rationale to name the semantic formula source and the degradation reason when confidence is not supported.

**Phase to address:** Phase 4: Operator Formula And Byte Modeling.

### Pitfall 14: Sidecar Parsing Lags Sidecar Emission

**What goes wrong:** New fields are emitted in v2 artifacts but parser round-trip tests do not validate them, so tooling silently drops evidence.

**Prevention strategy:** Every sidecar field addition needs `to_dict` and `from_dict` round-trip tests, including invalid payload tests for required machine-verifiable fields.

**Phase to address:** Phase 5: Artifact And Score Integration.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Phase 1: Scope, Claim Contract, And Acceptance Gates | Paper-aligned is read as paper-equivalent | Lock no-claim wording, preserve `amd-native-derived`, and add public-contract scan tests before feature work |
| Phase 2: Extractor Frontend And Coverage Semantics | FX/AST fallback hides skipped work | Record extraction provenance, skipped/unsupported operations, frontend failures, and coverage denominator evidence |
| Phase 3: Family-Specific Semantic Extraction | Shape/rank heuristics replace semantic axes | Require family-specific attributes and degrade when required axes/dtypes/layouts are unknown |
| Phase 4: Operator Formula And Byte Modeling | Taxonomy-only families inflate coverage | Promote a family only with formula, byte buckets, confidence rules, rationale, and positive/negative golden tests |
| Phase 5: Aggregate Bound, Artifact, And Score Eligibility | Partial evidence produces normal scores | Gate scored/degraded/unscored states on unsupported, inexact, coverage, hardware model, and baseline evidence |
| Phase 6: Documentation, Guardrails, And Closure | Derived artifacts leak into public contracts or claims | Re-run CLI/schema/trace guardrails, docs claim scans, sidecar round-trip tests, and scope checklist |

## Minimum Milestone Gates

| Gate | Required Evidence |
|------|-------------------|
| No overclaiming | Docs and tests state v1.10 is ROCm AMD-native derived SOLAR derivation, not paper benchmark parity, NVIDIA Blackwell/B200 equivalence, leaderboard readiness, or new hardware validation |
| No partial-score inflation | Unsupported extracted or skipped operations force `unscored`; inexact or provisional evidence forces `degraded`; suite reports expose scored and unscored counts |
| Shape/dtype correctness | Golden formula-input tests cover non-square, batched, broadcasted, mixed-dtype, and missing-axis cases for each promoted family |
| FX/AST robustness | Equivalent references across function, method, operator, module, helper, and fallback forms produce stable family/confidence outcomes or explicit warnings |
| Public contract stability | Canonical definition/workload/trace schemas and primary CLI help remain free of derived SOLAR fields |
| Sidecar auditability | Every v1.10 sidecar field round-trips and has machine-verifiable status, confidence, coverage, rationale, and warning evidence |

## Sources

- arXiv 2603.19173 abstract, "SOL-ExecBench: Speed-of-Light Benchmarking for Real-World GPU Kernels Against Hardware Limits", submitted 2026-03-19: https://arxiv.org/abs/2603.19173
- Local project scope and v1.10 exclusions: `.planning/PROJECT.md`
- Local milestone history and v1.9 delivered contracts: `.planning/MILESTONES.md`
- BoundGraph extraction, FX/AST fallback, op families, tensor metadata: `src/sol_execbench/core/scoring/amd_bound_graph.py`
- Operator estimate formulas, byte accounting, unsupported estimates: `src/sol_execbench/core/scoring/amd_bound_estimates.py`
- AMD SOL v2 sidecar status, coverage, warnings, parser: `src/sol_execbench/core/scoring/amd_sol_v2.py`
- AMD-native score warnings, claim level, score eligibility, suite counts: `src/sol_execbench/core/scoring/amd_score.py`
- Score interpretation guardrail: `src/sol_execbench/core/scoring_guardrails.py`
- Public contract guardrails: `tests/sol_execbench/test_public_contract_guardrails.py`
