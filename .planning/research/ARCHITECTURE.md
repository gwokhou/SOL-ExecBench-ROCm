# Architecture Research: v1.10 Paper-Aligned SOLAR Derivation

**Project:** SOL ExecBench ROCm Port  
**Domain:** AMD-native SOL/SOLAR bound derivation sidecar pipeline  
**Researched:** 2026-05-23  
**Overall confidence:** HIGH for repository integration boundaries, MEDIUM for paper-alignment details because only the arXiv abstract was used as the paper baseline.

## Executive Summary

v1.10 should integrate paper-aligned SOLAR derivation by extending the existing scoring sidecar pipeline, not by changing the benchmark execution path. The current architecture already has the correct backbone: canonical evaluation produces trace JSONL, optional dataset/reporting code derives AMD SOL v2 sidecars from `Definition` and `Workload`, and AMD-native score reports consume those sidecars plus measured trace timing and baseline latency. That shape preserves the project constraint that canonical trace JSONL, public schemas, and primary `sol-execbench` CLI behavior remain stable.

The paper baseline from arXiv 2603.19173 is that SOLAR computes analytical hardware-grounded Speed-of-Light bounds, and SOL Score measures how much of the gap between a release-defined scoring baseline and the hardware SOL bound a candidate closes. For this ROCm milestone, the architecture should therefore focus on automatic derivation evidence and coverage evidence: richer workload graph extraction, family-specific estimates, hardware-bound sidecars, and guarded score eligibility. It should not add 124-model/235-problem extraction, new real-hardware validation, or a hosted leaderboard.

The recommended integration point is a new internal SOLAR derivation layer inside `src/sol_execbench/core/scoring/` that sits between `amd_bound_graph.py` and `amd_sol_v2.py`. `amd_bound_graph.py` should continue to own workload-bound graph IR extraction, but it needs family-specific extractors and normalization helpers. `amd_bound_estimates.py` should remain the estimate facade but delegate to per-family estimator modules. `amd_sol_v2.py` should remain the stable sidecar builder/parser and aggregate coverage gate, with additive fields only if required for derivation coverage. `amd_score.py` should stay a consumer of sidecar aggregate state and must not duplicate derivation logic.

## Current Architecture Relevant To SOLAR

```text
Canonical benchmark path:
definition/workload/solution/config
  -> ProblemPackager
  -> isolated eval_driver subprocess
  -> canonical Trace JSONL

Derived scoring path:
Definition + Workload + AmdHardwareModel
  -> build_bound_graph()
  -> estimate_bound_work()
  -> build_amd_sol_bound_v2_artifact()
  -> score_amd_native_trace_workload()
  -> optional dataset suite report and sidecars
```

The current derived path is already separated from the execution path:

| Existing component | Current role | v1.10 role |
| --- | --- | --- |
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | Builds `BoundGraph` from `Definition.reference` and concrete `Workload`, using torch.fx first and AST fallback. | Extend into richer SOLAR workload graph extraction, while preserving `BoundGraph.to_dict()` compatibility or evolving it additively. |
| `src/sol_execbench/core/scoring/amd_bound_estimates.py` | Converts `BoundGraphNode` objects into FLOP/byte/movement estimates for common families. | Become the dispatcher to family-specific estimators; avoid growing one monolithic estimator file. |
| `src/sol_execbench/core/scoring/amd_sol_v2.py` | Builds/parses AMD SOL v2 sidecars with graph, estimates, per-op bounds, aggregate status, warnings, and coverage. | Remain the stable artifact boundary for SOLAR derivation evidence and score eligibility. |
| `src/sol_execbench/core/scoring/amd_score.py` | Computes guarded AMD-native scores from trace timing, baseline latency, and SOL artifacts. | Consume aggregate sidecar status only; do not inspect extractor internals. |
| `src/sol_execbench/core/scoring/amd_hardware_models.py` | Loads strict AMD hardware models with validation/confidence metadata. | Continue as hardware input contract; no new hardware validation claims in v1.10. |
| `scripts/run_dataset.py` | Optional dataset runner builds AMD SOL v2 sidecars and AMD score reports. | Add optional SOLAR coverage/derivation outputs here, preserving default benchmark CLI behavior. |

## Recommended Component Boundaries

### 1. SOLAR Extraction Layer

Add focused modules under `src/sol_execbench/core/scoring/`:

```text
solar_extraction.py          # orchestration: Definition + Workload -> BoundGraph
solar_patterns.py            # reusable graph pattern definitions
solar_shape.py               # workload-bound shape, dtype, axis, and broadcast evidence helpers
solar_coverage.py            # extraction/estimate coverage summaries and gates
```

`amd_bound_graph.py` can either import these helpers or be gradually split so existing public imports continue to work:

```python
def build_bound_graph(definition: Definition, workload: Workload) -> BoundGraph:
    return derive_solar_bound_graph(definition, workload)
```

This keeps the existing integration surface stable while making room for family-specific extraction. The important boundary is that extraction consumes only canonical problem inputs, especially `Definition.reference`, declared input/output schemas, axes, and concrete `Workload`. It must not consume solution code or measured trace latency, because SOLAR bounds are analytical targets independent of candidate performance.

### 2. Family-Specific Estimator Layer

Split estimator logic by operation family:

```text
solar_estimates/
  __init__.py                # estimate_bound_work dispatcher
  gemm.py
  attention.py
  moe.py
  convolution.py
  ssm_mamba.py
  embedding_positional.py
  normalization.py
  pointwise.py
  movement.py
```

`OperatorWorkEstimate` should remain the common estimate object. Additive evidence fields should be routed through existing `formula_kind`, `formula`, `formula_inputs`, `warnings`, `confidence`, and `rationale` first. Only add new serialized fields when they are needed for machine-verifiable coverage and cannot be represented in the existing fields.

Recommended new family responsibilities:

| Family | Extraction responsibility | Estimate responsibility | Confidence default |
| --- | --- | --- | --- |
| Attention | Detect QK, scale/mask/softmax, PV, GQA/MQA grouping, causal/window masks when visible in reference. | FLOPs for QK and PV, bytes for Q/K/V/O and mask/read-write movement, intermediate logits/probability evidence. | SUPPORTED only when dimensions and attention structure are explicit; otherwise INEXACT. |
| MoE | Detect top-k routing, expert projection groups, token dispatch/combine, expert count and capacity when inferable. | Routed GEMM FLOPs, routing/dispatch bytes, combine bytes, sparse activation evidence. | INEXACT unless top-k, expert shapes, and route cardinality are explicit. |
| Convolution | Detect `conv1d/2d/3d`, stride, padding, dilation, groups, batch/channel/spatial dimensions. | Standard convolution FLOPs and input/filter/output byte traffic. | SUPPORTED when shape and parameters are explicit. |
| SSM/Mamba | Detect scan/selective state update patterns, causal recurrence, projection pre/post steps. | Separate projection FLOPs from recurrence/scan estimates, with explicit recurrence warnings. | INEXACT by default; SUPPORTED only for recognized formulas with complete state dimensions. |
| Embedding/positional | Detect embedding lookup, gather, rotary/positional transform, index tensors. | Mostly memory/read traffic with low FLOP transforms for positional operations. | INEXACT if index cardinality or reuse is unknown. |
| Linear projection | Keep GEMM-compatible path but preserve semantic family for attention/MLP/MoE projections. | Delegate to GEMM formula with projection context in `formula_inputs`. | SUPPORTED when GEMM dims are known. |

### 3. Coverage Evidence Layer

The current `AmdSolV2CoverageSummary` counts total, supported, inexact, unsupported operations and confidence by family. v1.10 should extend coverage evidence so reports can distinguish "recognized but degraded" from "not extracted".

Recommended coverage model, preferably additive in the v2 sidecar:

```text
coverage_summary:
  total_ops
  supported_ops
  inexact_ops
  unsupported_ops
  op_family_counts
  confidence_counts_by_family
  worst_confidence
  extraction:
    source: torch.fx | ast | mixed | failed
    pattern_hits_by_family: {...}
    missing_pattern_families: [...]
    unsupported_operator_names: [...]
  estimate:
    estimated_node_ids: [...]
    unestimated_node_ids: [...]
    degraded_node_ids: [...]
```

If schema stability is a concern, add this as a new `derivation_coverage` field in `AmdSolBoundV2Artifact` rather than changing existing coverage fields. The parser should require it only after a schema version bump. For v1.10, a schema bump to `sol_execbench.amd_sol_bound.v3` is justified only if the roadmap requires strict parsing of new coverage fields. Otherwise, keep v2 and make coverage additive only in nested dictionaries that existing code tolerates.

### 4. Score Boundary

`amd_score.py` should remain intentionally narrow:

```text
Trace measured latency + baseline latency + sidecar aggregate SOL bound
  -> SOL Score or unscored/degraded warning
```

The score layer should not inspect attention/MoE/convolution details. It should rely on:

| Sidecar state | Score behavior |
| --- | --- |
| `aggregate_bound.status == "scored"` | Score can be computed if trace and baseline timing are complete. |
| `aggregate_bound.status == "degraded"` | Score may compute but must retain degraded/provisional warnings. |
| `aggregate_bound.status == "unscored"` | Score must be `None`; warnings must explain unsupported or missing evidence. |

This preserves the current guardrail that partial SOLAR coverage cannot silently inflate AMD-native scores.

## Data Flow For v1.10

```text
Inputs:
  Definition.reference
  Definition input/output schemas
  Workload axes
  AmdHardwareModel
  Optional canonical Trace JSONL and scoring baseline

Derivation:
  1. Build workload-bound tensor declarations from Definition + Workload.
  2. Trace reference with torch.fx when possible.
  3. Fall back to AST extraction for unsupported dynamic trace cases.
  4. Normalize graph nodes into paper-aligned families.
  5. Apply family-specific pattern extraction for attention, MoE, convolution,
     SSM/Mamba, embedding/positional, linear projection, and existing families.
  6. Estimate FLOPs, read/write bytes, intermediate bytes, movement bytes,
     formula inputs, confidence, and rationale per node.
  7. Convert estimates plus hardware model into per-op compute/memory SOL bounds.
  8. Aggregate operation bounds and coverage into scored/degraded/unscored state.
  9. Write sidecar evidence and optional suite coverage report.

Scoring:
  10. Combine canonical Trace latency, release baseline latency, and aggregate
      SOL bound through existing AMD-native score report code.
```

Critical separation:

```text
SOLAR derivation never changes:
  - definition.json schema
  - workload.jsonl schema
  - solution.json schema
  - Trace JSONL schema
  - eval_driver behavior
  - default sol-execbench CLI output
```

## New vs Modified Components

### New Components

| Component | Purpose | Build priority |
| --- | --- | --- |
| `solar_shape.py` | Shared shape/dtype/axis/broadcast/intermediate evidence helpers. | 1 |
| `solar_patterns.py` | Declarative operation-pattern recognizers for paper-aligned families. | 2 |
| `solar_extraction.py` | Orchestrates fx/AST extraction and family normalization. | 2 |
| `solar_estimates/attention.py` | Attention-specific analytical work and bytes. | 3 |
| `solar_estimates/convolution.py` | Convolution work/byte formulas. | 3 |
| `solar_estimates/embedding_positional.py` | Embedding, gather, RoPE/positional movement and FLOPs. | 3 |
| `solar_estimates/moe.py` | Routing, expert projection, dispatch/combine estimates. | 4 |
| `solar_estimates/ssm_mamba.py` | Selective scan/state-space evidence and degraded estimates. | 4 |
| `solar_coverage.py` | Coverage classification, missing-family evidence, aggregate gates. | 5 |
| Optional `scripts/derive_solar_bounds.py` | Batch derivation without running hardware evaluation. | 6 |

### Modified Components

| Component | Modification | Boundary |
| --- | --- | --- |
| `amd_bound_graph.py` | Delegate to SOLAR extraction helpers; add family-specific node attributes; keep existing `build_bound_graph` import path. | Internal scoring only. |
| `amd_bound_estimates.py` | Turn into dispatcher or compatibility facade over `solar_estimates`. | Existing `estimate_bound_work(graph)` remains. |
| `amd_sol_v2.py` | Add derivation coverage evidence and stricter aggregate gates. | Sidecar-only schema, no canonical schemas. |
| `amd_score.py` | Possibly add warning mapping for new coverage statuses. | No derivation internals. |
| `scripts/run_dataset.py` | Add opt-in coverage sidecar/report output, possibly build sidecars even without score report. | No default CLI behavior changes. |
| Tests under `tests/sol_execbench/` | Golden graph, estimator, sidecar parser, score guardrail, and public-contract tests. | CPU-friendly where possible. |
| Docs | Explain derived SOLAR evidence and non-claims. | Must preserve no-leaderboard/no-B200-equivalence language. |

## Suggested Phase Order

1. **Derivation Contract And Golden Fixtures**
   - Define the v1.10 SOLAR derivation contract: input sources, output sidecar fields, confidence states, coverage gates, and non-goals.
   - Add small golden fixtures for attention, convolution, embedding/positional, MoE, and SSM/Mamba reference snippets.
   - Rationale: the rest of the milestone needs stable expected evidence before estimators grow.

2. **Extraction Infrastructure**
   - Extract shape/dtype/axis helpers from `amd_bound_graph.py`.
   - Add pattern recognizers and family-specific node attributes.
   - Preserve `build_bound_graph(definition, workload)`.
   - Rationale: richer families need graph evidence before numerical formulas are credible.

3. **High-Confidence Estimator Families**
   - Implement convolution, embedding/positional, and linear projection refinements first.
   - Extend attention QK/softmax/PV derivation for explicit tensor shapes.
   - Rationale: these are formula-stable and give immediate coverage gains with lower ambiguity.

4. **Degraded Complex Families**
   - Implement MoE and SSM/Mamba with explicit degraded defaults.
   - Require unsupported or inexact states when routing cardinality, expert shapes, recurrence dimensions, or scan semantics are not machine-verifiable.
   - Rationale: these are important paper-aligned families but most likely to overclaim.

5. **Sidecar Coverage And Score Guards**
   - Add derivation coverage evidence to AMD SOL sidecars.
   - Tighten aggregate status rules so unsupported extraction blocks scoring, and degraded estimates stay visibly provisional.
   - Update `amd_score.py` only for warning propagation.
   - Rationale: coverage evidence is the core v1.10 deliverable and protects reports from partial derivation misuse.

6. **Dataset Runner Integration And Docs**
   - Add opt-in sidecar/coverage reporting in `scripts/run_dataset.py`, independent of real hardware validation.
   - Document how to produce derivation coverage without claiming new hardware validation.
   - Rationale: operational integration should happen after the sidecar contract is stable.

7. **Public Contract Guardrails**
   - Add tests proving canonical trace JSONL, public schemas, CLI help/default output, and evaluation isolation did not change.
   - Add static docs/tests for deferred scope: no 124-model extraction, no new real-hardware validation, no hosted leaderboard.
   - Rationale: this milestone is easy to accidentally expand beyond its intended boundary.

## Patterns To Follow

### Pattern: Sidecar-Only Derivation

**What:** SOLAR derivation produces evidence artifacts separate from canonical evaluation traces.  
**When:** Any new graph, coverage, formula, or bound evidence is needed.  
**Example:**

```python
artifact = build_amd_sol_bound_v2_artifact(
    definition,
    workload,
    hardware_model,
    hardware_model_ref="default_amd_hardware_models.gfx1200",
)
score = score_amd_native_trace_workload(trace, artifact, baseline_artifact=baseline)
```

### Pattern: Confidence-First Estimation

**What:** Every extracted node must produce either supported, inexact, or unsupported evidence with rationale.  
**When:** Estimator cannot prove the full analytical formula from reference/workload structure.  
**Example:** MoE top-k routing without explicit expert cardinality should produce an `INEXACT` estimate or `UNSUPPORTED` estimate, not a silent dense GEMM approximation.

### Pattern: Stable Facade, Internal Split

**What:** Keep existing public internal call sites like `build_bound_graph` and `estimate_bound_work`, but move implementation into focused modules.  
**When:** Refactoring `amd_bound_graph.py` and `amd_bound_estimates.py` for family-specific logic.  
**Benefit:** Existing dataset/reporting paths keep working while the implementation becomes testable by family.

## Anti-Patterns To Avoid

### Anti-Pattern: Putting SOLAR Fields In Trace JSONL

**Why bad:** Trace JSONL is the canonical evaluation output. Adding derivation evidence there changes public behavior and couples analytical bounds to runtime execution.

**Instead:** Write SOLAR evidence as AMD SOL sidecars and suite reports.

### Anti-Pattern: Solution-Aware Bounds

**Why bad:** SOLAR bounds are hardware-grounded analytical targets. Reading candidate solution code to derive bounds would make the target mutable and reward-hackable.

**Instead:** Derive from `Definition`, `Workload`, and hardware model only.

### Anti-Pattern: Dense Fallback For Sparse/MoE/SSM

**Why bad:** Treating unknown sparse routing or recurrence as dense supported work can inflate or deflate SOL bounds while appearing complete.

**Instead:** Emit degraded or unscored evidence with explicit missing dimensions and unsupported operator names.

### Anti-Pattern: Hardware Validation By Derivation

**Why bad:** Better analytical coverage is not new hardware validation.

**Instead:** Keep hardware model validation statuses unchanged and preserve CDNA 3/CDNA 4 deferral language.

## Scalability Considerations

| Concern | Small fixtures | Dataset-scale local runs | Future full paper-scale extraction |
| --- | --- | --- | --- |
| Extraction cost | In-process fx/AST per workload is fine. | Cache parsed reference and per-axis shape helpers by definition/workload UUID. | Add batch derivation command with resumable sidecar writes. |
| Sidecar size | Full graph evidence is useful. | Sidecars can grow; keep report summaries separate from full per-workload evidence. | Consider compressed artifacts or manifest indexing if needed. |
| Testability | Golden unit tests per family. | Contract tests for dataset runner opt-in outputs. | Add fixture packs only after extraction scope expands beyond v1.10. |
| Hardware assumptions | Packaged `gfx1200` model. | Hardware model ref stays explicit in every sidecar. | Add new model artifacts only with separate validation evidence. |

## Roadmap Implications

The roadmap should build from contracts to extraction to estimates to reporting. Starting with sidecar coverage gates is tempting, but coverage without richer family extraction would only repackage current partial modeling. Starting with complex families like MoE or SSM is also risky because their formulas require strong missing-evidence behavior.

Recommended milestone phase structure:

1. Contract and fixtures.
2. Extraction/shape infrastructure.
3. Stable high-confidence families: convolution, embedding/positional, linear projection, attention core.
4. Complex degraded families: MoE and SSM/Mamba.
5. Sidecar coverage and score guardrails.
6. Dataset-runner/docs integration.
7. Public-contract and no-claim guardrails.

## Research Flags For Later Phases

| Topic | Why it needs deeper validation | Suggested phase |
| --- | --- | --- |
| Attention pattern matching | Reference code may express attention through matmul, einsum, `scaled_dot_product_attention`, masking, or custom reshapes. | Extraction infrastructure and attention estimator phases. |
| MoE routing formulas | Sparse token routing can depend on runtime index distributions that are not visible from static shapes alone. | Complex degraded families phase. |
| SSM/Mamba formulas | Selective scan structure may not be reliably inferable from arbitrary Python reference code. | Complex degraded families phase. |
| Sidecar schema version | Additive v2 fields may be enough, but strict parser guarantees may require v3. | Sidecar coverage phase. |
| Paper parity terminology | The paper targets NVIDIA Blackwell and CUDA; ROCm reports must say AMD-native derived evidence, not B200/SOLAR equivalence. | Docs and guardrails phase. |

## Sources

- Repository project context: `.planning/PROJECT.md` (HIGH confidence for current milestone scope and constraints).
- Repository architecture: `docs/ARCHITECTURE.md` (HIGH confidence for canonical CLI, driver, and trace boundaries).
- Existing extraction and IR: `src/sol_execbench/core/scoring/amd_bound_graph.py` (HIGH confidence).
- Existing estimator facade: `src/sol_execbench/core/scoring/amd_bound_estimates.py` (HIGH confidence).
- Existing sidecar contract: `src/sol_execbench/core/scoring/amd_sol_v2.py` (HIGH confidence).
- Existing score reports: `src/sol_execbench/core/scoring/amd_score.py` (HIGH confidence).
- Existing hardware model contract: `src/sol_execbench/core/scoring/amd_hardware_models.py` (HIGH confidence).
- Existing dataset integration: `scripts/run_dataset.py` (HIGH confidence).
- arXiv 2603.19173 abstract: `https://arxiv.org/abs/2603.19173` (MEDIUM confidence for paper baseline because only abstract-level claims were used).
