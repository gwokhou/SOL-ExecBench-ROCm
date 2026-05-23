# Phase 50: Degraded Complex Family Modeling - Research

**Researched:** 2026-05-23  
**Domain:** SOLAR derivation for MoE and SSM/Mamba-like complex families  
**Confidence:** HIGH for repository integration points; MEDIUM for exact heuristic names because implementation remains planner-controlled

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### MoE Scope
- Recognize MoE routing, top-k selection, expert projection, token dispatch,
  and combine patterns when those structures are visible from reference or
  workload evidence.
- Record dynamic routing evidence when expert cardinality, token-to-expert
  assignment, top-k semantics, or static routing cardinality is incomplete.
- Treat incomplete static routing metadata as `inexact` / `degraded`, not as
  scored evidence.
- Use `unsupported` / `unscored` when the pipeline cannot determine core MoE
  semantics or cannot distinguish MoE from unrelated indexing/control-flow
  patterns.

### SSM/Mamba Scope
- Recognize SSM/Mamba-like projection, depthwise convolution, scan or state
  update, gating, and output projection patterns when structurally visible.
- Record recurrence and state-update evidence separately from ordinary
  convolution, data movement, or linear projection evidence.
- Treat incomplete recurrence, scan, state carry, sequence-order, or gating
  semantics as degraded evidence.
- Use `unsupported` / `unscored` when state-update semantics are absent or the
  visible structure is too generic to justify an SSM/Mamba classification.

### Confidence Strategy
- Phase 50 is conservative by default. `supported` is allowed only when family,
  subroles, shapes, dtypes, axes, routing/state semantics, and formula/byte
  provenance are sufficiently visible.
- Common dynamic routing and recurrence cases should become
  `inexact` / `degraded` with explicit `missing_evidence` and stable warning
  prefixes.
- The implementation must never fabricate expert counts, token dispatch
  cardinality, top-k values, sequence state lengths, recurrence semantics, or
  scan behavior.

### Integration Boundary
- Reuse Phase 49 group-local formula, byte, and bound evidence in
  `SolarSemanticGroupEvidence`.
- Reuse existing `BoundGraph`, `OperatorWorkEstimate`, `SolarDerivationEvidence`,
  strict parser, confidence, and public guardrail patterns.
- Do not modify canonical `Definition`, `Workload`, `Trace`, primary
  `sol-execbench` CLI behavior, canonical trace JSONL, or AMD-native score
  eligibility.
- Do not add new framework dependencies and do not execute submitted candidate
  solution code.

### the agent's Discretion
- Exact helper names, formula-kind names, and warning prefixes are at the
  agent's discretion as long as they are deterministic, parseable, and follow
  existing scoring/test conventions.
- The planner may split MoE and SSM/Mamba work by family, by shared degraded
  evidence infrastructure, or by graph/estimate/sidecar layers.

### Deferred Ideas (OUT OF SCOPE)
- Sidecar coverage aggregation, score eligibility changes, and report guard
  integration remain deferred to Phase 51.
- Dataset-runner reporting closure and public documentation remain deferred to
  Phase 52.
- Paper-scale 124-model / 235-problem extraction, MI300X/CDNA validation,
  hosted leaderboard readiness, and NVIDIA Blackwell/B200 equivalence claims
  remain out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DERIVE-02 | Conservatively recognize MoE routing, top-k selection, expert projection, token dispatch, and combine patterns, with dynamic routing evidence when static cardinality is incomplete. | Use graph annotation for visible router/top-k/dispatch/expert/combine subroles, estimate helpers that only score static metadata, and MoE-specific confidence gates. [VERIFIED: `.planning/REQUIREMENTS.md`] |
| DERIVE-04 | Conservatively recognize SSM/Mamba-like projection, depthwise convolution, scan/state update, gating, and output projection patterns, with degraded evidence when recurrence semantics are incomplete. | Use SSM graph annotation that keeps scan/state evidence distinct from convolution and projection, then degrade when state or update semantics are missing. [VERIFIED: `.planning/REQUIREMENTS.md`] |
</phase_requirements>

## Summary

Phase 50 should extend the existing repository-local SOLAR pipeline rather than introduce a new analyzer. `OpFamily` already contains `moe` and `ssm_mamba`, while the graph node, estimate, and sidecar models already carry attributes, confidence, warnings, formula evidence, byte evidence, and bound evidence. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py:27`; `src/sol_execbench/core/scoring/solar_derivation.py:809`]

The primary recommendation is to add conservative family annotations in `amd_bound_graph.py`, family-specific estimate dispatch in `amd_bound_estimates.py`, and family-specific subrole/confidence gates in `solar_derivation.py`. Supported/scored evidence should be rare; degraded evidence is the normal outcome when routing cardinality or recurrence semantics are visible but incomplete. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py:71`; `src/sol_execbench/core/scoring/solar_derivation.py:330`]

**Primary recommendation:** Implement MoE and SSM/Mamba as degraded-first semantic groups using existing sidecar fields, with explicit `missing_evidence` and stable warning prefixes instead of guessed dynamic metadata. [VERIFIED: `.planning/phases/50-degraded-complex-family-modeling/50-CONTEXT.md`]

## Project Constraints (from AGENTS.md)

- Use Python 3.12+, Ruff style, `snake_case` functions/modules, `PascalCase` classes/models, and local patterns. [VERIFIED: `AGENTS.md`]
- Tests belong under `tests/sol_execbench/`, with existing markers for ROCm/hardware-sensitive tests. [VERIFIED: `AGENTS.md`]
- Do not commit caches, build output, downloaded datasets, credentials, proprietary kernels, Hugging Face tokens, or benchmark assets. [VERIFIED: `AGENTS.md`]
- Commit titles use `#<Issue Number> - <Commit Title>` and DCO sign-off. [VERIFIED: `AGENTS.md`]
- ROCm >= 7.0, RDNA 4 and CDNA 3, preserved benchmark semantics, no required NVIDIA dual backend, and passing migrated tests remain project constraints. [VERIFIED: `AGENTS.md`]
- Direct source edits are not part of this research task. [VERIFIED: user request]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| MoE family recognition | Scoring graph extraction | Tests/fixtures | The graph extractor owns family/subrole annotations on `BoundGraphNode.attributes`. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py:89`] |
| MoE degraded/static estimates | Scoring estimates | SOLAR sidecar | `estimate_bound_work()` dispatches per-node estimates before sidecar grouping. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py:66`] |
| SSM/Mamba family recognition | Scoring graph extraction | Tests/fixtures | Existing annotation functions promote visible multi-op structures after FX/AST extraction. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py:638`] |
| SSM/Mamba degraded/static estimates | Scoring estimates | SOLAR sidecar | Formula, byte, and bound evidence is built from `OperatorWorkEstimate`. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:850`] |
| Confidence/status decisions | SOLAR sidecar | Scoring estimates | `classify_solar_confidence()` maps missing evidence and estimate confidence to `scored`, `degraded`, or `unscored`. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:330`] |
| Public schema/CLI preservation | Tests/guardrails | Scoring sidecar | Phase 49 verification confirms sidecar fields stay internal and guardrails exist. [VERIFIED: `.planning/phases/49-high-confidence-family-modeling/49-VERIFICATION.md`] |

## Standard Stack

### Core

| Library/Module | Version | Purpose | Why Standard |
|----------------|---------|---------|--------------|
| Python stdlib dataclasses/AST | Python 3.12+ | Internal immutable evidence and AST fallback extraction | Existing scoring code already uses dataclasses and `ast` extraction. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py:1`] |
| PyTorch FX already imported opportunistically | Existing project dependency | Reference-visible graph tracing and shape propagation | `build_bound_graph()` first tries FX and falls back to AST without requiring candidate execution. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py:262`] |
| Existing `sol_execbench.core.scoring` modules | local | Graph, estimates, SOLAR sidecar, and AMD hardware model conversion | Phase 49 wired group-local formula/byte/bound evidence through these modules. [VERIFIED: `.planning/phases/49-high-confidence-family-modeling/49-VERIFICATION.md`] |

### Supporting

| Module | Purpose | When to Use |
|--------|---------|-------------|
| `tests/sol_execbench/fixtures/solar_derivation/*.json` | Contract anchors for positive, degraded, and unsupported behavior | Use fixture expectations to choose missing evidence and warning prefixes. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/moe_degraded_dynamic_routing.json:10`] |
| `tests/sol_execbench/test_solar_derivation_family_modeling.py` | End-to-end sidecar behavior tests | Add MoE and SSM/Mamba positive/degraded/unsupported tests here. [VERIFIED: `tests/sol_execbench/test_solar_derivation_family_modeling.py`] |
| `tests/sol_execbench/test_amd_bound_graph.py` | Graph annotation tests | Add recognition/subrole/attribute tests here. [VERIFIED: `tests/sol_execbench/test_amd_bound_graph.py`] |
| `tests/sol_execbench/test_amd_bound_estimates.py` | Estimate formula/byte/confidence tests | Add estimate-level degraded and static metadata tests here. [VERIFIED: `tests/sol_execbench/test_amd_bound_estimates.py`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Existing FX/AST/local scoring stack | ONNX, MLIR, Dynamo, sympy, networkx | New framework dependencies are explicitly out of scope. [VERIFIED: `.planning/REQUIREMENTS.md`] |
| Sidecar-only evidence | Canonical schema fields | Canonical `Definition`, `Workload`, `Trace`, CLI, and trace JSONL changes are prohibited. [VERIFIED: `.planning/phases/50-degraded-complex-family-modeling/50-CONTEXT.md`] |
| Static-only degraded modeling | Candidate solution execution | Candidate execution for derivation is explicitly prohibited. [VERIFIED: user request; `.planning/REQUIREMENTS.md`] |

**Installation:** No package installation. [VERIFIED: user request]

## Package Legitimacy Audit

No external packages are recommended or installed for this phase. [VERIFIED: user request]

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| None | n/a | n/a | n/a | n/a | n/a | No install |

**Packages removed due to slopcheck [SLOP] verdict:** none. [VERIFIED: no packages]  
**Packages flagged as suspicious [SUS]:** none. [VERIFIED: no packages]

## Architecture Patterns

### System Architecture Diagram

```text
Definition + Workload
  -> build_bound_graph()
     -> FX trace when available
     -> AST fallback when FX unavailable
     -> annotate complex family graph
        -> MoE visible pieces: router -> top_k/dynamic route -> dispatch -> expert_projection -> combine
        -> SSM visible pieces: input_projection -> depthwise_convolution -> scan/state_update -> gating -> output_projection
  -> estimate_bound_work()
     -> family-specific static/degraded estimates
     -> unsupported estimate when core semantics are absent
  -> derive_solar_derivation_evidence()
     -> semantic group subroles
     -> formula_evidence / byte_evidence / bound_evidence
     -> missing_evidence + warning_prefixes
  -> internal sidecar only
```

This mirrors the current Phase 49 architecture: graph annotations feed estimates, and estimates feed group-local evidence. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:809`]

### Recommended Project Structure

```text
src/sol_execbench/core/scoring/
├── amd_bound_graph.py       # add MoE and SSM/Mamba annotation helpers
├── amd_bound_estimates.py   # add MoE and SSM/Mamba estimate helpers
└── solar_derivation.py      # add subroles and confidence gates

tests/sol_execbench/
├── test_amd_bound_graph.py
├── test_amd_bound_estimates.py
├── test_solar_derivation_family_modeling.py
├── test_solar_derivation_evidence.py
└── test_public_contract_guardrails.py
```

### Pattern 1: MoE Recognition

**What:** Promote a graph segment to `OpFamily.MOE` only when at least router plus one downstream MoE-specific role is visible; taxonomy-only names are unsupported. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/moe_unsupported_taxonomy_only.json`]

**Conservative subroles:**
- `router`: linear/projection or callable whose output feeds routing scores. [ASSUMED]
- `top_k`: `topk`, `argmax`, or explicit `k`/`top_k` attribute when statically visible. [ASSUMED]
- `dispatch`: gather/index/dispatch-like movement using router-selected expert ids. [ASSUMED]
- `expert_projection`: linear/GEMM expert computation over dispatched tokens or expert weights. [ASSUMED]
- `combine`: weighted combine/scatter/add of expert outputs back to token order. [ASSUMED]

**Supported/scored gate:** Require `router`, `top_k`, `dispatch`, `expert_projection`, and `combine`, plus static `tokens`, `hidden`, `experts`, and `top_k` evidence. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/moe_positive.json`]

**Degraded gate:** If router/dispatch/expert/combine are visible but `top_k`, token-to-expert assignment, or static cardinality is missing, emit `inexact` / `degraded`, `missing_evidence=("route:top_k", "route:static_cardinality", ...)`, and warnings including `inexact_operator:moe_dynamic_routing` and `aggregate_degraded:moe`. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/moe_degraded_dynamic_routing.json:10`]

**Unsupported gate:** If only an opaque MoE call/name is visible, emit `unsupported` / `unscored`, no fabricated subroles, and warnings including `unsupported_operator:moe_taxonomy_only` and `aggregate_unscored:moe`. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/moe_unsupported_taxonomy_only.json`]

### Pattern 2: SSM/Mamba Recognition

**What:** Promote a graph segment to `OpFamily.SSM_MAMBA` only when sequence projection/conv/scan-state evidence is visible; do not classify ordinary depthwise convolution or linear projection alone as SSM/Mamba. [VERIFIED: `.planning/phases/50-degraded-complex-family-modeling/50-CONTEXT.md`]

**Conservative subroles:**
- `input_projection`: linear/GEMM projection feeding the sequence-state path. [ASSUMED]
- `depthwise_convolution`: convolution node with `groups == channels` or explicit depthwise evidence. [VERIFIED: Phase 49 convolution metadata exists in `src/sol_execbench/core/scoring/amd_bound_estimates.py:367`]
- `scan`: recognized selective scan / scan-like call name or explicit state update chain. [ASSUMED]
- `state_update`: explicit recurrence/update formula metadata, state tensor, or state dimension. [ASSUMED]
- `gating`: activation or elementwise multiply that gates scan output. [ASSUMED]
- `output_projection`: linear/GEMM projection after scan/gate. [ASSUMED]

**Supported/scored gate:** Require `input_projection`, `depthwise_convolution`, `scan`, `state_update`, `gating`, and `output_projection`, plus static `sequence`, `hidden`, and `state` evidence. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_positive.json`]

**Degraded gate:** If projection/conv/scan are visible but state dimension or update formula is missing, emit `inexact` / `degraded`, `missing_evidence=("shape:state", "recurrence:update_formula", ...)`, and warnings including `inexact_operator:ssm_missing_recurrence` and `aggregate_degraded:ssm_mamba`. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_degraded_missing_recurrence.json:10`]

**Unsupported gate:** If the scan is opaque and no machine-verifiable recurrence contract exists, emit `unsupported` / `unscored`, `missing_evidence=("subrole:recognized_scan", "shape:state", "recurrence:update_formula")`, and warnings including `unsupported_operator:ssm_custom_scan` and `aggregate_unscored:ssm_mamba`. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_unsupported_custom_scan.json:10`]

### Pattern 3: Degraded Evidence Emission

**What:** Reuse `SolarSemanticGroupEvidence` fields exactly: `required_evidence`, `missing_evidence`, `warning_prefixes`, `formula_evidence`, `byte_evidence`, and `bound_evidence`. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:208`]

**How:**
- Estimates may carry formulas with empty `formula_inputs` only when the formula is a known family formula but required dynamic metadata is missing. [VERIFIED: convolution degraded behavior in `src/sol_execbench/core/scoring/amd_bound_estimates.py:375`]
- Byte evidence must sum only visible tensor bytes; missing shape or dtype should produce zero for that tensor plus `inexact_bytes:*` warnings, not guessed bytes. [VERIFIED: `tests/sol_execbench/test_amd_bound_estimates.py`]
- Bound evidence can still be emitted for degraded estimates, because Phase 49 already converts every estimate into sidecar-local bound evidence and carries confidence on the bound record. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:856`]
- Family-specific aggregate warnings should be added for MoE and SSM/Mamba, matching the fixture prefixes. [VERIFIED: fixture warning prefixes cited above]

### Anti-Patterns to Avoid

- **Taxonomy-only promotion:** A name containing "moe" or "mamba" is not enough to produce scored or degraded complex-family evidence. [VERIFIED: fixture unsupported cases]
- **Dynamic metadata fabrication:** Do not invent `top_k`, expert count, dispatch count, state length, or recurrence formula from tensor names or common model defaults. [VERIFIED: Phase 50 context]
- **Score eligibility drift:** Do not route degraded complex-family evidence into AMD-native score eligibility changes; Phase 51 owns score guard integration. [VERIFIED: Phase 50 context]
- **Convolution overreach:** Depthwise convolution remains convolution unless connected to scan/state/gating evidence. [VERIFIED: Phase 50 context]
- **Opaque scan scoring:** A custom `opaque_scan(x)` without state/update metadata must be unscored. [VERIFIED: `tests/sol_execbench/fixtures/solar_derivation/ssm_mamba_unsupported_custom_scan.json`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Public evidence schema | New public fields | Existing internal SOLAR sidecar dataclasses | Canonical schemas and CLI behavior must stay unchanged. [VERIFIED: Phase 50 context] |
| Graph IR | Separate MoE/SSM graph model | `BoundGraphNode.attributes` and `OpFamily` | Existing graph nodes already serialize attributes and confidence. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_graph.py:89`] |
| Confidence system | New confidence vocabulary | `EstimateConfidence` values `supported`, `inexact`, `unsupported` | Existing sidecar status mapping depends on these values. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:396`] |
| Byte/bound math plumbing | Duplicate SOL conversion | Existing `OperatorWorkEstimate` and `_bound_evidence_for_estimates()` | Phase 49 already converts estimates to bound evidence. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:856`] |
| Dynamic routing/scan execution | Running candidate/reference code for metadata | Static FX/AST/reference-visible evidence only | Candidate execution is prohibited and fixture loader tests avoid executing references. [VERIFIED: `tests/sol_execbench/test_solar_derivation_contract.py`] |

**Key insight:** Complex families need conservative semantic evidence, not better guessing. The implementation should make incomplete dynamic behavior observable as degraded or unsupported sidecar evidence. [VERIFIED: Phase 50 context]

## Common Pitfalls

### Pitfall 1: MoE False Positives From Generic Indexing
**What goes wrong:** Gather/scatter/topk-looking code is classified as MoE without visible router, expert, dispatch, and combine semantics. [ASSUMED]  
**Why it happens:** MoE routing reuses ordinary indexing/data movement operations. [ASSUMED]  
**How to avoid:** Require a minimum subrole set and explicit expert-axis evidence before `supported`; taxonomy-only or generic indexing should be unsupported. [VERIFIED: MoE fixtures]  
**Warning signs:** `moe` group has no `router` or `expert_projection` subrole. [VERIFIED: MoE unsupported fixture]

### Pitfall 2: Top-k and Cardinality Guessing
**What goes wrong:** The estimator fills `top_k=1` or `top_k=2` based on common model patterns. [ASSUMED]  
**Why it happens:** Common MoE examples often use top-1 or top-2, but Phase 50 forbids fabrication. [VERIFIED: Phase 50 context]  
**How to avoid:** Only set `top_k` from parsed call args/kwargs, workload axes, or definition-visible constants; otherwise add `route:top_k` and `route:static_cardinality` to `missing_evidence`. [VERIFIED: MoE degraded fixture]  
**Warning signs:** Formula inputs contain `top_k` while no source attribute records where it came from. [VERIFIED: existing provenance patterns require source details]

### Pitfall 3: Treating SSM/Mamba As Ordinary Conv + Linear
**What goes wrong:** A depthwise convolution plus projections is classified as SSM/Mamba even without scan/state update. [ASSUMED]  
**Why it happens:** Mamba-like blocks include projections and depthwise convolution, but the defining recurrence/scan evidence may be opaque. [ASSUMED]  
**How to avoid:** Require `scan` or `state_update` evidence for any SSM/Mamba group; otherwise leave nodes as convolution/linear or unsupported. [VERIFIED: SSM/Mamba fixtures]  
**Warning signs:** `ssm_mamba` group lacks `scan` and `state_update` subroles. [VERIFIED: SSM/Mamba fixtures]

### Pitfall 4: Degraded Evidence Without Parseable Warning Prefixes
**What goes wrong:** Tests cannot distinguish degraded MoE/SSM evidence from generic incomplete evidence. [VERIFIED: fixture loader validates warning prefixes]  
**Why it happens:** `classify_solar_confidence()` currently adds generic aggregate warnings and attention-specific aggregate warnings only. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:397`]  
**How to avoid:** Add family-specific aggregate prefixes for `moe` and `ssm_mamba` in confidence gates. [VERIFIED: fixture warning expectations]

## Code Examples

### Conservative MoE Estimate Shape

```python
# Source: existing estimate pattern in amd_bound_estimates.py
if missing:
    return OperatorWorkEstimate(
        node_id=node.node_id,
        op_family=OpFamily.MOE,
        op_name=node.op_name,
        formula_kind="moe_degraded_static_route_flops",
        formula="visible_router_flops+visible_expert_flops+visible_combine_bytes",
        formula_inputs={},
        flops=visible_flops,
        read_bytes=visible_read_bytes,
        write_bytes=visible_write_bytes,
        intermediate_bytes=visible_intermediate_bytes,
        movement_bytes=visible_dispatch_bytes,
        total_bytes=visible_total_bytes,
        confidence=EstimateConfidence.INEXACT,
        rationale="MoE route is visible but static routing cardinality is incomplete",
        axis_source=None,
        warnings=("inexact_operator:moe_dynamic_routing",),
    )
```

This follows the existing degraded convolution pattern: known formula kind, empty inputs when metadata is incomplete, visible bytes only, and `INEXACT` confidence. [VERIFIED: `src/sol_execbench/core/scoring/amd_bound_estimates.py:375`]

### SSM/Mamba Confidence Gate Shape

```python
# Source: existing family-specific confidence gate style in solar_derivation.py
required = {"input_projection", "depthwise_convolution", "scan", "state_update", "gating", "output_projection"}
missing.extend(f"ssm_mamba_subrole:{name}" for name in sorted(required - subroles))
if state_shape_missing or recurrence_formula_missing:
    missing.append("recurrence:update_formula")
    warnings.append("inexact_operator:ssm_missing_recurrence")
```

This mirrors existing attention/convolution family gates that add semantic missing evidence and warning prefixes before generic confidence classification. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py:1151`]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MoE and SSM/Mamba taxonomy existed but were deferred/unsupported | Phase 50 should add degraded-first recognition and evidence | v1.10 Phase 50 | Planner should add implementation tasks across graph, estimates, sidecar, and tests. [VERIFIED: Phase 50 context] |
| High-confidence families only | Group-local formula/byte/bound evidence exists for promoted families | Phase 49 | Complex families should reuse the same sidecar path. [VERIFIED: Phase 49 verification] |
| Generic aggregate degraded warnings | Family-specific warnings expected by fixtures for MoE and SSM/Mamba | Phase 47 fixtures | Add `aggregate_degraded:moe`, `aggregate_unscored:moe`, `aggregate_degraded:ssm_mamba`, and `aggregate_unscored:ssm_mamba`. [VERIFIED: fixture JSON] |

**Deprecated/outdated:** Treating complex families as silently unsupported is no longer sufficient for DERIVE-02 and DERIVE-04. [VERIFIED: `.planning/REQUIREMENTS.md`]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Specific MoE subrole heuristics can be detected from names like router/topk/dispatch/combine plus tensor flow. | MoE Recognition | Planner should require tests before relying on each heuristic. |
| A2 | Specific SSM/Mamba subrole heuristics can be detected from names like selective_scan, depthwise_conv, gate, in_proj, and out_proj plus tensor flow. | SSM/Mamba Recognition | Planner should keep unsupported fallback for opaque or differently named references. |
| A3 | Common MoE examples often use top-1/top-2, but implementation must not infer those values. | Common Pitfalls | Low risk because context already forbids fabrication; this is only explanatory. |

## Resolved Planning Decisions

1. **Formula-kind names are deterministic and family-scoped.**
   - What we know: formula/byte/bound evidence fields already exist and fixture expectations do not lock formula-kind names. [VERIFIED: Phase 50 context]
   - Decision: Use deterministic names and lock them in tests: `moe_static_route_flops`, `moe_dynamic_route_bytes`, `ssm_mamba_static_scan_flops`, and `ssm_mamba_degraded_scan_bytes`. Planners may add narrower operation-specific names only if they remain parseable and family-scoped.

2. **A scan call alone does not imply `state_update`.**
   - What we know: positive fixture expects `input_projection`, `depthwise_convolution`, `scan`, `state_update`, `gating`, and `output_projection`. [VERIFIED: SSM/Mamba positive fixture]
   - Decision: Do not infer `state_update` from an opaque or standalone scan call unless state shape and update parameters are visible. Otherwise emit degraded or unscored SSM/Mamba evidence with explicit missing recurrence/state evidence.

## Environment Availability

Step 2.6: SKIPPED. This phase has no new external runtime, CLI, service, database, or package dependency beyond the existing project test environment. [VERIFIED: user request]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest via `uv run pytest` [VERIFIED: AGENTS.md] |
| Config file | `pyproject.toml` contains pytest configuration and pytest dependencies. [VERIFIED: `pyproject.toml`] |
| Quick run command | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0` |
| Full suite command | `uv run pytest tests/` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DERIVE-02 | MoE static top-k emits supported/scored evidence with router/top-k/dispatch/expert/combine subroles | unit/integration | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0` | Partial; add tests |
| DERIVE-02 | MoE dynamic routing emits degraded evidence with `route:top_k` and `route:static_cardinality` missing | unit/integration | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0` | Partial; add tests |
| DERIVE-02 | MoE taxonomy-only case is unsupported/unscored | unit/integration | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0` | Partial; add tests |
| DERIVE-04 | SSM/Mamba static scan emits supported/scored evidence with projection/conv/scan/state/gate/out roles | unit/integration | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0` | Partial; add tests |
| DERIVE-04 | SSM/Mamba missing recurrence emits degraded evidence with state/update missing | unit/integration | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0` | Partial; add tests |
| DERIVE-04 | Opaque custom scan is unsupported/unscored | unit/integration | `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py -n 0` | Partial; add tests |

### Sampling Rate

- **Per task commit:** focused file tests for the touched layer. [VERIFIED: existing Phase 49 verification pattern]
- **Per wave merge:** `uv run pytest tests/sol_execbench/test_solar_derivation_family_modeling.py tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_v2.py -n 0`. [VERIFIED: Phase 49 gate]
- **Phase gate:** full focused gate plus Ruff over touched scoring/tests. [VERIFIED: AGENTS.md; Phase 49 verification]

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_amd_bound_graph.py` — add MoE/SSM graph annotation tests.
- [ ] `tests/sol_execbench/test_amd_bound_estimates.py` — add MoE/SSM estimate tests for supported, degraded, and unsupported cases.
- [ ] `tests/sol_execbench/test_solar_derivation_family_modeling.py` — add sidecar group tests matching Phase 47 fixtures.
- [ ] `tests/sol_execbench/test_solar_derivation_evidence.py` — add parser/round-trip expectations only if no schema fields are added; otherwise guard exact-key parsing for any new nested data.
- [ ] `tests/sol_execbench/test_public_contract_guardrails.py` — extend internal-field absence checks for any new sidecar-only field names.

## Security Domain

Security enforcement is enabled by default because `.planning/config.json` does not set `security_enforcement: false`. [VERIFIED: `.planning/config.json`]

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No authentication surface in this phase. [VERIFIED: phase scope] |
| V3 Session Management | no | No session surface in this phase. [VERIFIED: phase scope] |
| V4 Access Control | no | No access-control surface in this phase. [VERIFIED: phase scope] |
| V5 Input Validation | yes | Strict sidecar parser and fixture validators should reject malformed evidence. [VERIFIED: `src/sol_execbench/core/scoring/solar_derivation.py`; `tests/sol_execbench/solar_derivation_fixtures.py`] |
| V6 Cryptography | no | No cryptographic operation in this phase. [VERIFIED: phase scope] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Fabricated dynamic metadata inflates derived bounds | Tampering | Require source-backed formula inputs and explicit `missing_evidence` when metadata is incomplete. [VERIFIED: Phase 50 context] |
| Candidate solution execution during derivation | Elevation of Privilege | Keep derivation APIs built from canonical `Definition` and `Workload`; tests should assert candidate-free paths. [VERIFIED: Phase 50 context] |
| Internal sidecar evidence leaks into public schema/CLI | Information Disclosure | Extend public contract guardrails. [VERIFIED: Phase 49 verification] |
| Opaque custom functions masquerade as MoE/SSM | Tampering | Taxonomy-only and opaque-scan tests must remain unscored. [VERIFIED: fixtures] |

## Sources

### Primary (HIGH confidence)

- `.planning/ROADMAP.md` - Phase 50 scope and milestone boundaries.
- `.planning/REQUIREMENTS.md` - DERIVE-02, DERIVE-04, no new dependency, no candidate execution, no public schema changes.
- `.planning/phases/50-degraded-complex-family-modeling/50-CONTEXT.md` - locked phase decisions.
- `.planning/phases/49-high-confidence-family-modeling/49-VERIFICATION.md` - verified Phase 49 integration points and test gate.
- `src/sol_execbench/core/scoring/amd_bound_graph.py` - graph taxonomy, annotation style, FX/AST extraction.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - estimate dispatch and degraded formula/byte patterns.
- `src/sol_execbench/core/scoring/solar_derivation.py` - sidecar grouping, confidence, formula/byte/bound evidence, parser patterns.
- `tests/sol_execbench/fixtures/solar_derivation/*.json` - MoE and SSM/Mamba fixture expectations.
- `tests/sol_execbench/test_amd_bound_graph.py`, `test_amd_bound_estimates.py`, `test_solar_derivation_family_modeling.py`, `test_solar_derivation_evidence.py`, `test_public_contract_guardrails.py` - relevant test targets.

### Secondary (MEDIUM confidence)

- None used. Repository-local contracts were sufficient. [VERIFIED: no web/package research required]

### Tertiary (LOW confidence)

- Assumed heuristic names for recognizing router/top-k/dispatch/combine and selective scan/gating when the reference uses nonstandard helper names. [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new dependencies; existing modules and tests are verified in the repository.
- Architecture: HIGH - Phase 49 verification and current code show the graph -> estimate -> sidecar flow.
- Pitfalls: MEDIUM - fixture-backed risks are HIGH, but exact heuristic coverage for arbitrary references remains implementation-dependent.

**Research date:** 2026-05-23  
**Valid until:** 2026-06-22 for repository architecture; revisit sooner if Phase 51 changes score eligibility semantics.
