# Feature Landscape

**Domain:** Paper-aligned SOLAR automatic derivation for SOL ExecBench ROCm
**Milestone:** v1.10 论文级完整 SOLAR 自动推导
**Researched:** 2026-05-23
**Overall confidence:** HIGH for repo-local capability gaps and milestone scope; MEDIUM for exact paper-family priority beyond the arXiv abstract because v1.10 explicitly excludes full paper dataset extraction.

## Scope Decision

v1.10 should make the AMD SOL/SOLAR pipeline look honest at the derivation layer: given a SOL ExecBench definition and workload, the code should automatically derive richer hardware-bound evidence for paper-relevant kernel families, expose what was recognized, explain every FLOP/byte/movement estimate, and deterministically degrade to inexact or unscored states when coverage is incomplete. It should not claim that the ROCm port reproduces the original paper's full 124-model/235-problem benchmark, NVIDIA Blackwell target, or new AMD hardware validation.

The arXiv 2603.19173 abstract frames the paper baseline as analytically derived hardware-grounded SOL bounds from SOLAR, SOL Score reporting against those bounds, broad real-world kernel coverage across model families, and robustness guardrails for agentic optimizer evaluation. For this ROCm milestone, the actionable subset is automatic derivation and score-evidence honesty on AMD: broader extraction/modeling for attention, MoE, convolution, SSM/Mamba, embedding/positional, and linear projection patterns; stronger artifact coverage semantics; and guardrails that prevent partial derivation from looking complete.

Current code already has a useful v1.9 foundation:

- `BoundGraph` IR with tensor metadata, dependency edges, confidence, rationale, dynamic `torch.fx` tracing with AST fallback, and an `OpFamily` taxonomy that already names the paper-aligned families.
- Operator estimates for GEMM/BMM, linear projection via GEMM behavior, elementwise, activation, reduction, normalization, softmax, data movement, and dtype conversion.
- v2 AMD SOL sidecars with graph, estimate, op-bound, aggregate-bound, hardware-model, coverage, and warning evidence.
- AMD-native score reports that propagate degraded/unscored states and claim-level warnings.

The v1.10 feature line is therefore not "add SOLAR docs"; it is "turn taxonomy placeholders and simple local patterns into code-real automatic derivation paths, while making every unsupported gap machine-visible."

## Table Stakes

Features users and roadmap consumers should expect. Missing any of these leaves v1.10 incomplete.

| Feature | Why Expected | Complexity | Requirements Notes |
|---------|--------------|------------|--------------------|
| Automatic paper-family extraction coverage | The milestone goal says attention, MoE, convolution, SSM/Mamba, embedding/positional, and linear projection need explicit extraction and estimation paths rather than taxonomy-only placeholders. | High | Extend `build_bound_graph()` classification and extraction beyond the current simple call list. Each new family needs at least one deterministic fixture that produces a non-unsupported `OpFamily` node from reference/workload structure. |
| Attention derivation path | Attention is central to language, diffusion, vision, audio, video, and hybrid model workloads named by the paper baseline. | High | Recognize Q/K/V projections, QK matmul, scale, mask/add, softmax, AV matmul, and output projection where present. Emit either a compound `attention` node with child evidence or a linked subgraph with `attention_group_id`; unsupported masks/dropout/control flow must degrade explicitly. |
| MoE derivation path | Mixture-of-experts is a paper-relevant real-world model family and has distinct routing/top-k/scatter/gather/expert GEMM behavior. | High | Recognize router logits, top-k/argmax-like selection, gating weights, gather/scatter/dispatch, expert projection/GEMM, combine. Mark routing capacity, sparse dispatch, or dynamic expert counts as `INEXACT` unless fully resolved. |
| Convolution derivation path | Vision/diffusion workloads require convolution coverage; treating convolution as unsupported makes broad-kernel claims hollow. | High | Recognize `conv1d`, `conv2d`, `conv3d`, module call names if trace exposes them, stride/padding/dilation/groups, kernel shape, batch/channel/output dimensions. Estimate FLOPs and bytes with grouped/depthwise-specific formulas. |
| SSM/Mamba derivation path | The current taxonomy names SSM/Mamba, and the v1.10 scope explicitly calls it out. | High | Recognize scan/selective-scan style compositions conservatively: projections, depthwise conv, recurrent/scan movement, gating, elementwise update, output projection. If true recurrence/scan length cannot be derived, retain visible inexact/unsupported evidence instead of collapsing to elementwise only. |
| Embedding and positional derivation path | Embedding/positional work is common in model frontends and can be memory-bound. | Medium | Recognize embedding lookup, gather/index_select/take, positional add/rotary/sinusoidal patterns when statically visible. Model bytes and movement separately from FLOPs; unsupported dynamic indexing remains visible. |
| Linear projection as a first-class family | Current estimates model `linear` with GEMM-like behavior, but v1.10 needs user-visible family evidence, not just generic GEMM. | Medium | Preserve `linear_projection` in graph and estimates with projection-specific formula kind and rationale while using GEMM math where valid. Capture bias add as explicit elementwise or projection attribute. |
| Compound-family grouping evidence | Paper-aligned derivation should show workload structure, not only a flat list of primitive calls. | High | Add stable grouping metadata such as `group_id`, `parent_family`, `subrole`, or equivalent fields inside derived sidecars. Do not mutate canonical `Trace` or public schemas. |
| Rich formula evidence for every new family | SOLAR derives hardware-grounded bounds; users need inspectable formula inputs, not opaque labels. | High | Each new estimator must emit `formula_kind`, `formula`, `formula_inputs`, FLOPs, read/write/intermediate/movement/total bytes, confidence, rationale, and warnings. Unsupported placeholders are not enough for table-stakes families. |
| Shape, dtype, and axis provenance | Automatic derivation is only reviewable if estimates can be traced back to definition/workload axes and tensor metadata. | Medium | Carry resolved dimensions, tensor IDs, axis sources, dtype widths, and source expressions through new family estimates. Missing metadata should downgrade confidence and add warnings. |
| Deterministic unsupported/inexact degradation | Honest paper alignment depends on not fabricating precision. | Medium | Known semantics with incomplete metadata become `INEXACT`; unknown operations or unresolved key effects become `UNSUPPORTED`; aggregate bounds become `degraded` or `unscored` by deterministic policy. |
| Family-aware score eligibility | AMD-native reports must protect users from treating partial SOLAR derivation as complete. | Medium | Extend v2 coverage/aggregate evidence so reports can say which required families were modeled, inexact, unsupported, or absent. Unsupported evidence should prevent SOL Score computation for that workload unless policy explicitly allows guarded degraded scoring. |
| Machine-verifiable complete/degraded/unscored states | The milestone asks for complete, degraded, and unscored states that are machine-verifiable. | Medium | Keep `aggregate_bound.status`, `scored`, `reason`, `worst_confidence`, family counts, and warnings parseable. Add tests for round-trip parsing and score behavior for each state. |
| Golden derivation fixtures | Roadmap requirements need code-real acceptance criteria. | Medium | Add small deterministic fixtures for attention, MoE, convolution, SSM/Mamba, embedding/positional, linear projection, and a mixed unsupported case. Assert graph families, formulas, warnings, coverage, aggregate status, and score behavior. |
| Public contract isolation | SOLAR derivation is a sidecar/scoring subsystem, not a canonical benchmark output change. | Low | Keep new evidence in derived AMD SOL v2+ artifacts or compatible sidecars. Guardrails should prove `definition.json`, `workload.jsonl`, `solution.json`, canonical trace JSONL, and primary CLI behavior remain stable. |
| Robustness guardrail propagation | The paper baseline mentions sandboxing, clock locking, cache clearing, subprocess isolation, and static checks; the ROCm port already has many runtime guardrails. v1.10 must ensure derivation results do not weaken them. | Medium | Do not create bypass paths that import untrusted solution code during derivation. Preserve existing reward-hack/static-analysis and score-claim warnings. Add derivation-specific guardrails against unsupported-op omission and overclaim language. |

## Differentiators

Features that would make v1.10 more credible than merely adding formulas for a few calls. These are valuable if table-stakes items are underway.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Pattern-level recognizers over raw call matching | Captures model-family semantics such as attention and MoE even when implemented as primitive PyTorch operations. | High | Prefer deterministic graph-pattern passes over adding many aliases to `_CALL_CLASSIFIERS`. This is the clearest path from v1.9 local operator modeling to SOLAR-like derivation. |
| Hierarchical sidecar view | Lets reviewers inspect both compound families and primitive child operations. | Medium | Useful for attention/MoE/SSM where a single family contains matmul, softmax, movement, and elementwise subroles. |
| Coverage inventory by required family | Makes the roadmap and users see exactly how paper-aligned coverage changed. | Medium | Report per-workload and suite-level counts for attention, MoE, convolution, SSM/Mamba, embedding/positional, linear projection, and primitive families. |
| Confidence thresholds configurable for derived reports | Supports strict research workflows without changing derivation output. | Medium | Example: allow reports to require all table-stakes families supported, or permit inexact families with warnings. Keep default conservative. |
| Family-specific rationale templates | Improves auditability and test stability. | Low | Rationale should state why attention/MoE/conv/SSM was recognized and which assumptions made the estimate inexact. |
| Minimal internal derivation CLI or helper API | Helps maintainers inspect a single definition/workload without running a full benchmark. | Medium | Only add if it stays derived/internal or opt-in; do not alter primary `sol-execbench` defaults. |
| Negative golden fixtures | Proves honesty by showing malformed/ambiguous patterns become inexact or unscored. | Low | Include dynamic control flow, unknown custom calls, unresolved shapes, dynamic indexing, and partial attention patterns. |

## Anti-Features

Features to explicitly not build in v1.10.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Original paper 124-model / 235-problem extraction | User explicitly excluded dataset extraction and curation from v1.10. Building it would expand scope and delay derivation correctness. | Use targeted synthetic and small real-shaped fixtures that exercise derivation families. |
| New real-hardware validation claims | User explicitly excluded new real-hardware validation. Current validation remains RDNA 4-scoped from prior work; CDNA 3/MI300X and CDNA 4 remain deferred. | Keep hardware validation status in artifacts and warnings. Do not promote unvalidated architectures. |
| Hosted leaderboard or submission service | User explicitly excluded a hosted leaderboard. | Keep reports local and derived. |
| NVIDIA Blackwell/B200 equivalence | The paper targets NVIDIA Blackwell GPUs; this ROCm port produces AMD-native derived evidence. | Continue using `amd-native-derived` claim language and no-equivalence guardrails. |
| Docs-only SOLAR alignment | The user asked for code-real capabilities, not claims. | Require extractor, estimator, artifact, score-report, and test changes for each table-stakes family. |
| Taxonomy-only family support | `OpFamily` already contains attention/MoE/convolution/SSM/embedding labels; labels without extraction and estimates do not satisfy v1.10. | Every claimed family needs recognizer coverage, estimate evidence, and golden tests. |
| Silent primitive fallback for compound patterns | Flattening attention or MoE into primitive matmul/elementwise nodes hides missing paper-family derivation. | Preserve primitive evidence but add grouping/family evidence or explicit "compound not recognized" warnings. |
| Optimistic zero-cost unsupported nodes | Zeroing unsupported work can make SOL bounds too low and scores misleading. | Unsupported evidence must affect aggregate status and score eligibility. |
| Cache/fusion/performance inference from reference code alone | Reference semantics do not prove implementation cache behavior or fusion strategy. | Keep estimates semantic/logical with confidence labels; profiler timing remains separate evidence. |
| Canonical trace/schema mutation | Existing public contracts are intentionally stable. | Emit all SOLAR derivation data as sidecars or opt-in derived reports. |
| Importing or executing submitted solution code for derivation | Derivation should be based on reference/workload structure, not untrusted candidate code. | Keep derivation on definitions/reference functions and existing safe harness boundaries. |

## Feature Categories for REQUIREMENTS.md

Use these categories directly when drafting v1.10 requirements.

### DERIVE: Automatic Extraction

| Requirement Category | Capability |
|----------------------|------------|
| DERIVE-ATTN | Detect attention structures from reference/workload graph evidence, including Q/K/V projections, score matmul, softmax, value matmul, and output projection where present. |
| DERIVE-MOE | Detect MoE routing, expert selection, dispatch/gather/scatter, expert projection/GEMM, gating, and combine patterns where statically visible. |
| DERIVE-CONV | Detect convolution calls/modules with dimensional parameters and derive output dimensions from workload axes. |
| DERIVE-SSM | Detect SSM/Mamba-like projection, depthwise convolution, scan/update, gating, and output projection structures conservatively. |
| DERIVE-EMBED | Detect embedding/positional/gather/rotary-like memory-bound structures. |
| DERIVE-LINEAR | Preserve linear projection as a first-class family with projection-specific evidence. |
| DERIVE-GROUP | Attach compound-family grouping/subrole metadata without breaking existing graph nodes and edges. |

### MODEL: Work And Bound Evidence

| Requirement Category | Capability |
|----------------------|------------|
| MODEL-FORMULA | Emit formula kind, formula string, and formula inputs for each newly supported family. |
| MODEL-BYTES | Emit read, write, intermediate, movement, and total byte evidence with dtype-aware widths. |
| MODEL-PROVENANCE | Link formulas and bytes to tensor IDs, source expressions, axes, shapes, dtypes, and attribute sources. |
| MODEL-CONFIDENCE | Apply supported/inexact/unsupported confidence consistently by family and metadata completeness. |
| MODEL-BOUND | Convert new family estimates into compute, memory, limiting resource, per-op SOL bound, and aggregate bound evidence. |

### REPORT: Honesty And Score Semantics

| Requirement Category | Capability |
|----------------------|------------|
| REPORT-COVERAGE | Report family-aware coverage counts and worst confidence for required paper families and primitive families. |
| REPORT-STATUS | Preserve machine-verifiable `scored`, `degraded`, and `unscored` states with parseable reasons. |
| REPORT-SCORE | Prevent SOL Score computation when required SOL bound evidence is unscored; warn when evidence is degraded. |
| REPORT-CLAIMS | Keep AMD-native derived language and no NVIDIA/B200/SOLAR/leaderboard/hardware-validation equivalence warnings. |
| REPORT-REFS | Preserve evidence references for trace, timing, SOL bound sidecar, baseline, and hardware model. |

### TEST: Verification Fixtures

| Requirement Category | Capability |
|----------------------|------------|
| TEST-GOLDEN-FAMILIES | Golden fixtures for attention, MoE, convolution, SSM/Mamba, embedding/positional, and linear projection. |
| TEST-DEGRADATION | Fixtures proving partial or ambiguous derivation becomes inexact, degraded, unsupported, or unscored as appropriate. |
| TEST-ROUNDTRIP | Sidecar parse/serialize tests for new fields and coverage summaries. |
| TEST-SCORE | AMD-native score tests for complete, degraded, and unscored SOLAR evidence. |
| TEST-CONTRACT | Guardrails proving canonical schemas, trace JSONL, and primary CLI behavior are unchanged. |

## Supported Family Targets

| Family | Current v1.9 State | v1.10 Target | Confidence Target |
|--------|--------------------|--------------|-------------------|
| Attention | Taxonomy label exists; primitives like GEMM/softmax/elementwise can be modeled separately. | Recognize and group attention structure; estimate QK/AV/projection FLOPs, softmax passes, mask/add bytes, and movement. | `INEXACT` by default; `SUPPORTED` only for simple fully resolved dense cases if tests prove formula completeness. |
| MoE | Taxonomy label exists; no explicit estimator. | Recognize routing/expert/dispatch/combine patterns and expose sparse/dynamic assumptions. | Usually `INEXACT`; `UNSUPPORTED` for unresolved dynamic routing effects. |
| Convolution | Taxonomy label exists; no explicit estimator. | Estimate conv FLOPs/bytes for common 1D/2D/3D grouped/depthwise shapes. | `SUPPORTED` for fully resolved standard conv formulas; `INEXACT` for ambiguous module metadata. |
| SSM/Mamba | Taxonomy label exists; no explicit estimator. | Recognize conservative SSM/Mamba subgraphs and model projections, depthwise conv, scan/update, gating, and movement. | `INEXACT`; `UNSUPPORTED` for opaque custom scan kernels. |
| Embedding/positional | Taxonomy label exists; no explicit estimator. | Model memory-bound lookup/gather/positional/rotary patterns with low FLOPs and explicit bytes. | `INEXACT`; `SUPPORTED` only for static resolved lookup byte formulas. |
| Linear projection | Recognized and estimated through GEMM-like path. | Preserve first-class projection family, bias evidence, and projection-specific formulas. | `SUPPORTED` for fully resolved dense projection; `INEXACT` when bias/broadcast metadata is incomplete. |
| Existing primitive families | Implemented for GEMM, BMM, elementwise, activation, reduction, normalization, softmax, data movement, dtype conversion. | Keep stable and use as child evidence for compound families. | Preserve existing confidence behavior unless richer formulas justify upgrades. |

## Feature Dependencies

```text
Reference/workload graph evidence
  -> Primitive extraction
  -> Compound-family pattern recognition
  -> Family/subrole grouping metadata
  -> Family-specific work estimates
  -> Per-op and aggregate AMD SOL bounds
  -> Coverage/status/score eligibility
  -> AMD-native score report warnings

Shape/dtype/axis provenance
  -> Formula inputs
  -> Byte buckets
  -> Confidence decisions
  -> Degraded/unscored status reasons

Golden family fixtures
  -> Sidecar round-trip tests
  -> Score status tests
  -> Public contract guardrails
  -> Requirements acceptance evidence
```

## MVP Recommendation

Prioritize:

1. Add compound-family metadata and coverage/status semantics first, so every later recognizer has a stable place to report evidence.
2. Implement linear projection, convolution, and embedding/positional next because their formulas are tractable and unblock visible breadth.
3. Implement attention recognition as grouped primitive evidence with conservative formulas; this is the most important paper-alignment signal.
4. Implement MoE and SSM/Mamba conservatively with explicit inexact/unsupported degradation for dynamic parts.
5. Wire all new family evidence through v2 sidecars, AMD-native score eligibility, round-trip parsing, and golden tests.

Defer:

- Full original-paper extraction pipeline: out of scope.
- New RDNA/CDNA hardware validation: out of scope.
- Exact implementation-level fusion/cache modeling: not derivable from semantic reference code alone.
- Hosted leaderboard behavior: out of scope.

## Sources

- `https://arxiv.org/abs/2603.19173` - paper abstract baseline: hardware-grounded SOL bounds from SOLAR, broad real-world kernel families, SOL Score, and robustness guardrails. Confidence: HIGH for abstract-level scope.
- `.planning/PROJECT.md` - v1.10 milestone goal, explicit target families, and deferred dataset/hardware/leaderboard scope. Confidence: HIGH.
- `.planning/MILESTONES.md` - v1.9 delivered foundation and known gaps. Confidence: HIGH.
- `src/sol_execbench/core/scoring/amd_bound_graph.py` - current BoundGraph IR, taxonomy, FX/AST extraction, and unsupported evidence behavior. Confidence: HIGH.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - current estimator coverage and explicit unsupported status for attention, MoE, SSM/Mamba, convolution, and embedding/positional families. Confidence: HIGH.
- `src/sol_execbench/core/scoring/amd_sol_v2.py` - v2 sidecar aggregate status, coverage summary, warning, and parse/serialize semantics. Confidence: HIGH.
- `src/sol_execbench/core/scoring/amd_score.py` - AMD-native score warning and unscored/degraded behavior. Confidence: HIGH.
- `tests/sol_execbench/test_amd_bound_graph.py` and `tests/sol_execbench/test_amd_bound_estimates.py` - current golden coverage for graph extraction and estimates, plus proof that named paper families are taxonomy-only/unsupported today. Confidence: HIGH.
