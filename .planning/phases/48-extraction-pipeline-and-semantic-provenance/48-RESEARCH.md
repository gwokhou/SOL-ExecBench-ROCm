# Phase 48: Extraction Pipeline And Semantic Provenance - Research

**Researched:** 2026-05-23  
**Domain:** Internal SOLAR derivation evidence, semantic provenance, and deterministic confidence  
**Confidence:** HIGH for local architecture and validation strategy; MEDIUM for exact semantic grouping heuristics because family-specific formulas are deferred.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Evidence Shape And Storage
- Store new derivation evidence only in internal sidecar/scoring artifacts; do
  not add it to canonical trace JSONL, public Pydantic benchmark schemas, or
  primary CLI default output.
- Record tensor shape, dtype, semantic axes, source kind, source detail, and
  confidence rationale for each derivation node.
- Represent compound-family grouping with a new internal semantic
  group/evidence dataclass that can reference existing `BoundGraphNode` and
  estimate data without rewriting the v1.9 `BoundGraph` core.
- Encode missing evidence with structured `missing_evidence` fields plus stable
  warning prefixes so reports remain readable and tests remain machine
  verifiable.

#### Extraction Inputs And Boundaries
- Derive evidence from `Definition.reference`, `Workload.axes`,
  `Workload.inputs`, existing shape/dtype resolution, and FX/AST-visible
  structure only. Do not execute candidate solution code for derivation.
- When static information is incomplete, produce `inexact` or `unsupported`
  evidence with explicit missing fields instead of guessing paper-scale values.
- Do not implement family-specific formulas in Phase 48; keep attention,
  convolution, linear projection, embedding, MoE, and SSM/Mamba formula logic
  for Phases 49 and 50.
- Reuse the existing FX-first and AST-fallback extraction pattern, and tag
  provenance sources such as `fx`, `ast`, `workload`, and `definition`.

#### Confidence Rules
- Reuse the existing `supported`, `inexact`, and `unsupported` confidence
  vocabulary and map those states to `scored`, `degraded`, and `unscored`.
- Require visible family, subroles, shape, dtype, semantic axes, and
  formula/source provenance before marking evidence as `supported`.
- Mark evidence as `inexact` when family or subroles are visible but axes,
  mask, routing, padding, recurrence, dtype, byte evidence, or related metadata
  is incomplete.
- Mark evidence as `unsupported` when the pipeline cannot determine family or
  core subrole semantics, or when key dimensions or semantic evidence required
  for formulas are absent; unsupported evidence must carry missing evidence.

#### Phase 48 Test And Integration Contract
- The minimum deliverable is a parseable, serializable internal provenance
  evidence contract validated against Phase 47 fixtures for fields, states, and
  sidecar-only boundaries.
- Extend public contract guardrails to prove canonical schemas, trace JSONL,
  and primary CLI options do not expose Phase 48 derivation evidence fields.
- Provide internal derivation evidence and parse/serialize helpers only; do not
  change AMD-native score eligibility or promote the evidence into score
  reports until Phase 51.
- Split implementation planning in dependency order: evidence model and
  serialization, extractor plumbing, deterministic confidence rules, then
  fixture-driven guardrails.

### the agent's Discretion
- Exact class, module, and helper names are at the agent's discretion as long
  as they follow existing scoring/test conventions and remain internal.
- The planner may decide whether the initial evidence model lives in a new
  module or near existing AMD SOL v2 sidecar code, provided public schemas and
  primary CLI behavior remain unchanged.

### Deferred Ideas (OUT OF SCOPE)
- Family-specific formula modeling for high-confidence families is deferred to
  Phase 49.
- Conservative MoE and SSM/Mamba modeling is deferred to Phase 50.
- Sidecar coverage aggregation, AMD-native score eligibility changes, and
  score guard integration are deferred to Phase 51.
- Dataset-runner reporting closure and public documentation are deferred to
  Phase 52.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DERIVE-07 | The derivation pipeline emits compound-family grouping, subrole, and provenance metadata without mutating canonical trace JSONL or public benchmark schemas. | Add an internal semantic evidence artifact that references `BoundGraphNode` IDs and fixture expectations while extending public contract guardrails. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| MODEL-03 | Formula and byte evidence carries tensor shape, dtype, semantic-axis, and extraction-source provenance. | Reuse `BoundTensor.shape`, `BoundTensor.dtype`, node attributes such as `dim` and `axis_source`, and add explicit semantic-axis/source records in the Phase 48 evidence model. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_estimates.py] |
| MODEL-04 | The estimator applies deterministic supported, inexact, and unsupported confidence rules based on metadata completeness and recognized semantics. | Reuse `EstimateConfidence` and v2 aggregate status semantics, but keep Phase 48 output internal and sidecar-only. [VERIFIED: src/sol_execbench/core/scoring/amd_hardware_models.py] [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] |
</phase_requirements>

## Summary

Phase 48 should create a new internal semantic provenance layer over the existing bound graph and work-estimate pipeline. `BoundGraph`, `BoundGraphNode`, `BoundTensor`, and `OperatorWorkEstimate` already expose stable node IDs, tensor shape/dtype, operation family, extraction source attributes, warnings, and confidence vocabulary; Phase 48 should not rewrite those types as its first move. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_estimates.py]

The clean implementation path is a new internal module under `src/sol_execbench/core/scoring/`, likely `solar_derivation.py`, with frozen dataclasses and strict `to_dict()` / `from_dict()` helpers matching the style used by `amd_sol_v2.py`. The artifact should be parseable JSON-safe sidecar evidence, not a public benchmark schema field or primary CLI option. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

**Primary recommendation:** Build a sidecar-only `SolarDerivationEvidence` artifact from `Definition`, `Workload`, `BoundGraph`, and optional `OperatorWorkEstimate` records; validate it against Phase 47 fixtures before adding family-specific formulas. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: docs/internal/solar_derivation_contract.md]

## Project Constraints (from AGENTS.md)

- Source code belongs under `src/sol_execbench/`; tests belong under `tests/`, especially `tests/sol_execbench/` for package tests. [VERIFIED: AGENTS.md]
- Use Python 3.12+ and Ruff style; keep focused changes consistent with nearby modules. [VERIFIED: AGENTS.md]
- Use pytest, with small unit tests for schema and driver logic and integration tests when subprocess or GPU execution changes. [VERIFIED: AGENTS.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, or downloaded datasets. [VERIFIED: AGENTS.md]
- Preserve benchmark semantics and public schemas unless a ROCm-specific change is unavoidable. [VERIFIED: AGENTS.md]
- GSD workflow says repo edits should happen through a GSD command unless explicitly bypassed; this research was invoked as GSD phase research. [VERIFIED: AGENTS.md] [VERIFIED: gsd-sdk query init.phase-op 48]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Semantic evidence extraction | Python scoring layer | Data schema layer as read-only input | The extractor consumes `Definition` and `Workload` but must not mutate public Pydantic models. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: src/sol_execbench/core/data/definition.py] |
| Compound-family grouping | Python scoring layer | Existing bound graph IR | Grouping should reference `BoundGraphNode.node_id` and `OpFamily` instead of changing canonical graph core. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] |
| Tensor shape/dtype provenance | Existing bound graph IR | Semantic evidence sidecar | `BoundTensor` already stores shape, dtype, role, producer, and source; Phase 48 should annotate this with semantic axes and source details. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] |
| Confidence/status decisions | Python scoring layer | v2 artifact warning/status vocabulary | Existing `supported` / `inexact` / `unsupported` and `scored` / `degraded` / `unscored` semantics should be reused. [VERIFIED: docs/internal/solar_derivation_contract.md] [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] |
| Public contract preservation | Tests/guardrails | CLI and Pydantic models | Existing guardrails assert that derivation and v2 fields stay out of canonical models and CLI help. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |

## Existing Architecture

### Current Extraction Flow

```text
Definition.reference + Workload.axes/inputs
        |
        v
build_bound_graph()
        |
        +-- torch.fx trace + ShapeProp when possible
        |
        +-- AST fallback when FX tracing fails
        |
        v
BoundGraph(nodes, tensors, edges, warnings)
        |
        v
estimate_bound_work()
        |
        v
OperatorWorkEstimate records
        |
        v
build_amd_sol_bound_v2_artifact()
        |
        v
Internal v2 sidecar with aggregate status, coverage, warnings
```

- `build_bound_graph()` resolves input/output shapes from `Definition` and `Workload`, builds declared tensors, tries FX tracing with shape propagation, and falls back to AST parsing if FX tracing fails. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py]
- FX extraction executes `Definition.reference` to obtain the reference `run` function for tracing, but Phase 48 must not execute candidate solution code; this matches the phase boundary because the input is the canonical reference, not a submitted solution. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]
- AST fallback records dynamic control flow and unsupported calls as graph evidence instead of throwing away the case. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: tests/sol_execbench/test_amd_bound_graph.py]
- `BoundGraphNode.attributes` currently carries axis, movement, dtype target, and trace-source metadata without adding new dataclass fields for every kind of evidence. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: tests/sol_execbench/test_amd_bound_graph.py]
- `OperatorWorkEstimate` already serializes formula kind, formula text, formula inputs, FLOPs, byte buckets, confidence, rationale, axis source, movement kind, and warnings. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_estimates.py]
- `AmdSolBoundV2Artifact` is already a stable internal sidecar with strict parse/serialize helpers and aggregate status logic. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [VERIFIED: tests/sol_execbench/test_amd_sol_v2.py]

### Phase 47 Contract Inputs

- Fixture files under `tests/sol_execbench/fixtures/solar_derivation/*.json` define the target family, fixture class, source snippet, workload axes, expected subroles, required evidence, missing evidence, warning prefixes, and scope boundary. [VERIFIED: docs/internal/solar_derivation_contract.md] [VERIFIED: tests/sol_execbench/solar_derivation_fixtures.py]
- The fixture loader validates six target families: `attention`, `moe`, `convolution`, `ssm_mamba`, `embedding_positional`, and `linear_projection`. [VERIFIED: tests/sol_execbench/solar_derivation_fixtures.py]
- The fixture loader validates confidence values `supported`, `inexact`, and `unsupported`, and aggregate statuses `scored`, `degraded`, and `unscored`. [VERIFIED: tests/sol_execbench/solar_derivation_fixtures.py]
- Fixture references are contract snippets and the loader explicitly does not execute their text. [VERIFIED: tests/sol_execbench/test_solar_derivation_contract.py]

### Public Boundary

- `Definition`, `Workload`, and `Trace` public model dumps currently do not include SOLAR derivation evidence fields. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- The primary `sol-execbench --help` output is guarded against derived workflow options and v1.10 SOLAR derivation options. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- The v1 AMD SOL artifact is guarded against v2-only fields, so Phase 48 must avoid accidentally widening legacy artifacts. [VERIFIED: tests/sol_execbench/test_amd_sol_v2.py]

## Standard Stack

No new external package should be added for Phase 48. The required stack is already present:

| Component | Version / Source | Purpose | Why Standard |
|-----------|------------------|---------|--------------|
| Python | 3.12.13 available through `uv run python --version` | Implementation language | Repository requires Python 3.12+. [VERIFIED: AGENTS.md] [VERIFIED: uv run python --version] |
| pytest | 9.0.2 available through `uv run pytest --version` | Unit and contract tests | Repository uses pytest and configures markers in `pyproject.toml`. [VERIFIED: pyproject.toml] [VERIFIED: uv run pytest --version] |
| dataclasses | Python stdlib | Internal immutable sidecar records | Existing scoring sidecars use frozen dataclasses and `to_dict()` methods. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] |
| torch.fx | Existing project dependency via PyTorch | Reference graph extraction when possible | Existing `build_bound_graph()` uses `symbolic_trace` and `ShapeProp`. [VERIFIED: pyproject.toml] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] |
| ast | Python stdlib | Static fallback extraction | Existing bound graph extractor parses reference text with `ast.parse`. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] |

**Installation:** No package installation is recommended. [VERIFIED: .planning/REQUIREMENTS.md]

## Package Legitimacy Audit

Not applicable. Phase 48 should install no external packages and should use only the repository's current Python, PyTorch FX, AST, dataclass, and pytest stack. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: pyproject.toml]

## Recommended Implementation Strategy

### 1. Add A Separate Internal Evidence Module

Add a new module under `src/sol_execbench/core/scoring/`, such as `solar_derivation.py`, with frozen dataclasses for:

- `SolarEvidenceSource`: `kind` values such as `definition`, `workload`, `fx`, `ast`, `estimate`; `detail`; optional node/tensor references. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]
- `SolarTensorEvidence`: tensor ID/name, shape, dtype, semantic axes, source, and missing evidence. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py]
- `SolarSubroleEvidence`: subrole name, node IDs, tensor IDs, source, confidence, rationale, and missing evidence. [VERIFIED: docs/internal/solar_derivation_contract.md]
- `SolarSemanticGroupEvidence`: family, group ID, node IDs, subroles, confidence, status, missing evidence, warning prefixes, and rationale. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: tests/sol_execbench/solar_derivation_fixtures.py]
- `SolarDerivationEvidence`: schema version, derived flag, definition, workload UUID, groups, tensors, warnings, and optional fixture expectation links for tests. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [ASSUMED]

Keep these types internal and export them from `src/sol_execbench/core/scoring/__init__.py` only if a later internal caller needs package-level imports. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]

### 2. Build Evidence From Existing Graph And Estimate Objects

Implement a builder shaped like:

```python
def build_solar_derivation_evidence(definition: Definition, workload: Workload) -> SolarDerivationEvidence:
    graph = build_bound_graph(definition, workload)
    estimates = estimate_bound_work(graph)
    return derive_solar_evidence(definition, workload, graph, estimates)
```

This keeps the high-level call parallel to `build_amd_sol_bound_v2_artifact()` and makes tests easy to write. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [ASSUMED]

The lower-level `derive_solar_evidence()` should also accept prebuilt `BoundGraph` and estimates so tests can directly exercise edge cases without retracing references. [VERIFIED: tests/sol_execbench/test_amd_sol_v2.py] [ASSUMED]

### 3. Keep Grouping Shallow In Phase 48

Phase 48 should group visible nodes into semantic families and subroles without calculating final family formulas. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]

Recommended Phase 48 grouping scope:

| Family | Phase 48 Role | Evidence To Record |
|--------|---------------|--------------------|
| `linear_projection` | Strongest initial path because `OpFamily.LINEAR_PROJECTION` and GEMM estimates already exist. | Input/output/weight shape evidence, feature axes when visible, dtype, node IDs, confidence. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_estimates.py] |
| `attention` | Recognize only structural subroles visible in node sequence and source expressions. | Q/K/V-like inputs, matmul score, softmax axis, PV-like matmul, output projection if visible. [VERIFIED: docs/internal/solar_derivation_contract.md] [ASSUMED] |
| `convolution` | Record taxonomy/provenance stubs until formula phase if the existing graph does not classify conv calls. | Conv op/source expression, stride/padding/dilation attributes when statically visible, missing metadata otherwise. [VERIFIED: docs/internal/solar_derivation_contract.md] [ASSUMED] |
| `embedding_positional` | Record lookup/gather-like evidence from reference names/source expressions and tensor metadata only. | Index/table/output shape, dtype, missing index/static cardinality evidence. [VERIFIED: docs/internal/solar_derivation_contract.md] [ASSUMED] |
| `moe` | Record degraded or unsupported evidence from visible router/dispatch/expert/combine markers only. | Router, dispatch, expert projection, top-k/static-cardinality gaps. [VERIFIED: docs/internal/solar_derivation_contract.md] |
| `ssm_mamba` | Record degraded or unsupported evidence from visible scan/state/gate markers only. | Projection, depthwise conv, scan/state update, gating, recurrence gaps. [VERIFIED: docs/internal/solar_derivation_contract.md] |

Planner note: the [ASSUMED] heuristics above should be framed as fixture-driven minimal recognizers, not production-quality full model extraction.

### 4. Deterministic Confidence Rules

Use one pure helper, for example `classify_solar_confidence(group)`, with no filesystem, hardware, or candidate-solution dependency. [ASSUMED]

Rules to plan:

- `supported` + `scored`: family is recognized; required subroles are present; tensor shapes, dtype, semantic axes, source, and confidence rationale are complete; missing evidence is empty. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]
- `inexact` + `degraded`: family or core subroles are visible, but mask/routing/padding/recurrence/axis/dtype/byte evidence is incomplete; missing evidence is non-empty; warnings include stable `inexact_operator:` or `aggregate_degraded:` prefixes as applicable. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: tests/sol_execbench/solar_derivation_fixtures.py]
- `unsupported` + `unscored`: family or core subroles cannot be determined, key dimensions are absent, or static semantics are not derivable; missing evidence is non-empty; warnings include stable `unsupported_operator:` or `aggregate_unscored:` prefixes as applicable. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: tests/sol_execbench/solar_derivation_fixtures.py]

### 5. Serialization And Parser Rules

Mirror `amd_sol_bound_v2_from_dict()` style: require every top-level field, reject invalid schema versions, parse list/object/string fields strictly, and serialize tuples as lists. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py]

Recommended schema fields:

```json
{
  "schema_version": "sol_execbench.solar_derivation.v1",
  "derived": true,
  "definition": "name",
  "workload_uuid": "uuid",
  "groups": [],
  "tensors": [],
  "warnings": [],
  "source_boundary": {
    "canonical_trace_jsonl": false,
    "public_schema": false,
    "candidate_solution_execution": false
  }
}
```

The exact schema name is discretionary, but it should not reuse `sol_execbench.amd_sol_bound.v2` because this is semantic provenance, not a score-bound artifact. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [ASSUMED]

## Architecture Patterns

### Recommended Project Structure

```text
src/sol_execbench/core/scoring/
├── amd_bound_graph.py          # existing graph extraction, read-only integration
├── amd_bound_estimates.py      # existing formula/byte estimates, read-only integration
├── amd_sol_v2.py               # existing score-bound sidecar, unchanged except optional internal references
└── solar_derivation.py         # new internal semantic provenance contract

tests/sol_execbench/
├── test_solar_derivation_evidence.py      # new parser/serializer/confidence tests
├── test_solar_derivation_contract.py      # existing fixture contract tests
└── test_public_contract_guardrails.py     # extend for Phase 48 forbidden fields/options
```

[VERIFIED: AGENTS.md] [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [VERIFIED: tests/sol_execbench/test_solar_derivation_contract.py]

### Pattern: Evidence Over Failure

Unsupported or incomplete references should produce parseable evidence with `missing_evidence`, warnings, and `unsupported` or `inexact` confidence instead of exceptions, as long as the public `Definition` itself is valid. [VERIFIED: docs/internal/solar_derivation_contract.md] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py]

### Pattern: Sidecar-Only Public Boundary

Every new field name should be asserted absent from `Definition.model_dump()`, `Workload.model_dump()`, `Trace.model_dump()`, and primary CLI help. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

### Pattern: Stable Warning Prefixes

Use fixture-approved prefixes: `graph_warning:`, `estimate_warning:`, `inexact_operator:`, `unsupported_operator:`, `aggregate_degraded:`, and `aggregate_unscored:`. [VERIFIED: docs/internal/solar_derivation_contract.md] [VERIFIED: tests/sol_execbench/solar_derivation_fixtures.py]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Python reference parsing | A custom parser | `ast` and existing AST extractor patterns | The repository already uses AST fallback and tests unsupported control flow behavior. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] |
| Graph tracing | A new tracing engine | Existing `torch.fx` path in `build_bound_graph()` | FX trace + `ShapeProp` already populate node and tensor metadata. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] |
| JSON schema framework | New dependency | Frozen dataclasses plus strict parser helpers | Existing internal sidecars use this pattern and no new framework dependencies are in scope. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [VERIFIED: .planning/REQUIREMENTS.md] |
| Confidence vocabulary | New enum/status labels | `EstimateConfidence` plus fixture aggregate statuses | Contract and v2 sidecar already define the accepted vocabulary. [VERIFIED: docs/internal/solar_derivation_contract.md] [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] |
| Candidate behavior derivation | Executing submitted solutions | `Definition.reference`, `Workload.axes`, `Workload.inputs`, FX/AST-visible structure | Candidate solution execution is explicitly out of scope. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |

## Common Pitfalls

### Pitfall 1: Accidentally Expanding Public Schemas

**What goes wrong:** New evidence fields appear in `definition.json`, `workload.jsonl`, trace JSONL, or CLI help. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]  
**How to avoid:** Keep all Phase 48 types under scoring sidecars and add forbidden-field guardrails for `solar_derivation`, `semantic_groups`, `semantic_axes`, `source_detail`, and any chosen schema version string. [ASSUMED]

### Pitfall 2: Turning Fixture Expectations Into Formula Implementation

**What goes wrong:** Phase 48 starts calculating attention, convolution, MoE, or SSM/Mamba formulas. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]  
**How to avoid:** Limit Phase 48 to evidence shape, provenance, missing evidence, and confidence classification; leave formula fields as absent, placeholder, or sourced from existing generic estimates only. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]

### Pitfall 3: Treating Unknowns As Supported

**What goes wrong:** A family label or source-name match creates `supported` evidence even when required axes or subroles are absent. [VERIFIED: docs/internal/solar_derivation_contract.md]  
**How to avoid:** Require complete family, subrole, shape, dtype, axis, and source evidence before `supported`; otherwise use `inexact` or `unsupported` with explicit missing evidence. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]

### Pitfall 4: Mutating `BoundGraph` Too Broadly

**What goes wrong:** Phase 48 rewrites the v1.9 bound graph core and destabilizes existing v2 score tests. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: tests/sol_execbench/test_amd_sol_v2.py]  
**How to avoid:** Add a separate semantic evidence artifact that references node IDs and tensor IDs. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]

### Pitfall 5: Losing Determinism

**What goes wrong:** Evidence ordering depends on dict iteration, graph trace quirks, or unordered set traversal. [VERIFIED: tests/sol_execbench/test_amd_bound_graph.py] [VERIFIED: tests/sol_execbench/test_amd_sol_v2.py]  
**How to avoid:** Sort serialized tensors/groups where practical, deduplicate warnings in first-seen order like `_unique()`, and assert repeated builds serialize identically. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [ASSUMED]

## Validation Architecture

Nyquist validation is enabled because `.planning/config.json` has `workflow.nyquist_validation: true`. [VERIFIED: .planning/config.json]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 [VERIFIED: uv run pytest --version] |
| Config file | `pyproject.toml` [VERIFIED: pyproject.toml] |
| Quick run command | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x` [ASSUMED] |
| Fixture/guardrail run command | `uv run pytest tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0` [VERIFIED: tests/sol_execbench/test_solar_derivation_contract.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| Existing regression run command | `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_v2.py -n 0` [VERIFIED: tests/sol_execbench/test_amd_bound_graph.py] [VERIFIED: tests/sol_execbench/test_amd_sol_v2.py] |

### Phase Requirements To Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| DERIVE-07 | Evidence includes semantic groups, subroles, source kind/detail, node/tensor references, and does not mutate public schemas. | unit + guardrail | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_public_contract_guardrails.py -n 0 -x` | New evidence test needed; guardrail file exists. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| MODEL-03 | Tensor and formula/byte evidence carry shape, dtype, semantic axes, and extraction-source provenance. | unit | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_amd_bound_graph.py -n 0 -x` | New evidence test needed; bound graph test exists. [VERIFIED: tests/sol_execbench/test_amd_bound_graph.py] |
| MODEL-04 | Deterministic confidence maps complete evidence to `supported/scored`, incomplete visible evidence to `inexact/degraded`, and absent core semantics to `unsupported/unscored`. | unit + fixture matrix | `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py -n 0 -x` | New evidence test needed; fixture contract file exists. [VERIFIED: tests/sol_execbench/test_solar_derivation_contract.py] |

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_solar_derivation_evidence.py` should cover schema version, parse/serialize round trips, bad payload rejection, source records, tensor semantic axes, missing evidence, and deterministic warnings. [ASSUMED]
- [ ] Public guardrails should add the final Phase 48 evidence field names once implementation names are chosen. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] [ASSUMED]
- [ ] Fixture-driven tests should map at least one positive, one degraded, and one unsupported fixture through the new evidence contract before expanding all 18 fixtures. [VERIFIED: tests/sol_execbench/fixtures/solar_derivation] [ASSUMED]

### Sampling Rate

- Per task commit: run `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py -n 0 -x`. [ASSUMED]
- Per wave merge: run `uv run pytest tests/sol_execbench/test_solar_derivation_evidence.py tests/sol_execbench/test_solar_derivation_contract.py tests/sol_execbench/test_public_contract_guardrails.py -n 0`. [ASSUMED]
- Phase gate: run the above plus `uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_v2.py -n 0`. [ASSUMED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `uv` | Running tests and Python commands | yes | path `/home/guohao/.cargo/bin/uv` | none needed. [VERIFIED: which uv] |
| Python | Package runtime | yes | 3.12.13 | none needed. [VERIFIED: uv run python --version] |
| pytest | Validation | yes | 9.0.2 | none needed. [VERIFIED: uv run pytest --version] |
| Graphify context | Optional semantic graph lookup | no | disabled and `.planning/graphs/graph.json` absent | Continue with local files. [VERIFIED: ls .planning/graphs/graph.json] [VERIFIED: graphify status] |

**Missing dependencies with no fallback:** None found for Phase 48 research and planning. [VERIFIED: which uv] [VERIFIED: uv run pytest --version]

**Missing dependencies with fallback:** Graphify is unavailable, but the phase scope is sufficiently covered by local source, tests, requirements, and context files. [VERIFIED: graphify status]

## Security Domain

Security enforcement is enabled because `.planning/config.json` does not disable it. [VERIFIED: .planning/config.json]

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No authentication surface is changed. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |
| V3 Session Management | no | No session surface is changed. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |
| V4 Access Control | no | No service access boundary is changed. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |
| V5 Input Validation | yes | Strict parser helpers should validate sidecar payload shapes and reject invalid schema versions. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] |
| V6 Cryptography | no | No cryptographic behavior is changed. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |

Known threat patterns:

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Candidate solution execution during derivation | Elevation/Tampering | Derive only from `Definition.reference`, `Workload.axes`, `Workload.inputs`, and FX/AST-visible structure. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |
| Public artifact claim drift | Information integrity | Keep evidence sidecar-only and extend public contract guardrails. [VERIFIED: docs/internal/solar_derivation_contract.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| Malformed sidecar payload accepted as evidence | Tampering | Use strict from-dict parsing like `amd_sol_bound_v2_from_dict()`. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] |

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Fixture snippets do not all instantiate as full `Definition` objects because fixture JSON only supplies reference and axes, not full input/output schemas. | Planner may overpromise end-to-end fixture execution in Phase 48. | Build contract-level tests that compare evidence fields to fixture expectations, and use hand-authored `Definition`/`Workload` cases for graph extraction where needed. [VERIFIED: tests/sol_execbench/fixtures/solar_derivation/attention_positive.json] [ASSUMED] |
| Heuristic family recognition becomes too broad. | False `supported` evidence could contaminate later scoring phases. | Default ambiguous matches to `inexact` or `unsupported` and require missing evidence. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |
| Adding exports in `__init__.py` expands the perceived public API. | Internal contract may look public. | Prefer module-local imports in tests unless implementation needs package-level exports. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |
| Reusing `AmdSolBoundV2Artifact` for semantic provenance conflates score eligibility with derivation evidence. | Phase 51 boundaries blur early. | Create a separate schema version and do not feed Phase 48 evidence into AMD-native scoring. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] |
| FX path executes reference code during tracing setup. | Reference code with side effects could be problematic if expanded beyond trusted benchmark definitions. | Keep Phase 48 no-candidate-execution boundary explicit; do not execute fixture references in the loader; consider tests that use AST-only helper paths for fixture snippets. [VERIFIED: tests/sol_execbench/test_solar_derivation_contract.py] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] |

## Plan Suggestions

Recommended plan split:

1. **Evidence Model And Parser:** Add `solar_derivation.py` with schema constants, dataclasses, `to_dict()`, `solar_derivation_from_dict()`, and strict validation tests. This should complete the sidecar contract shape before extraction behavior exists. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [ASSUMED]
2. **Graph/Estimate Provenance Builder:** Add builder functions that consume `Definition`, `Workload`, `BoundGraph`, and `OperatorWorkEstimate`, then emit tensor/source evidence with shape, dtype, semantic axis placeholders, node IDs, and warning propagation. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: src/sol_execbench/core/scoring/amd_bound_estimates.py] [ASSUMED]
3. **Semantic Group And Confidence Rules:** Add deterministic family/subrole grouping helpers and confidence/status mapping. Start with minimal fixture-relevant recognition and conservative `inexact` / `unsupported` defaults. [VERIFIED: docs/internal/solar_derivation_contract.md] [ASSUMED]
4. **Fixture And Public Boundary Guardrails:** Add tests that compare evidence against Phase 47 fixtures and extend `test_public_contract_guardrails.py` for chosen Phase 48 field names/options. [VERIFIED: tests/sol_execbench/test_solar_derivation_contract.py] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

Planning should keep each task small enough to run focused tests without requiring ROCm hardware, Docker, or candidate solution execution. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: pyproject.toml]

## Code Examples

### Internal Dataclass Pattern

```python
@dataclass(frozen=True)
class SolarEvidenceSource:
    kind: str
    detail: str
    node_id: str | None = None
    tensor_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "node_id": self.node_id,
            "tensor_id": self.tensor_id,
        }
```

This mirrors the local `to_dict()` sidecar style, but the exact class names are discretionary. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] [ASSUMED]

### Strict Parser Pattern

```python
def _require_keys(payload: dict[str, object], required: set[str], *, source: str) -> None:
    for key in sorted(required):
        if key not in payload:
            raise ValueError(f"{source} missing required field: {key}")
```

This pattern is already used for AMD SOL v2 sidecar parsing and should be reused for Phase 48 evidence parsing. [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| v1 AMD SOL artifact with simpler graph/work estimates | v2 sidecar with bound graph, operator estimates, aggregate status, coverage, and warnings | v1.9, shipped 2026-05-23 | Phase 48 can layer semantic provenance over existing sidecar machinery. [VERIFIED: .planning/ROADMAP.md] [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py] |
| Fixture-free derivation expansion | Phase 47 golden fixture matrix first | Phase 47, complete 2026-05-23 | Phase 48 can validate evidence against known family/state expectations. [VERIFIED: .planning/ROADMAP.md] [VERIFIED: tests/sol_execbench/fixtures/solar_derivation] |
| Public schema changes for derived data | Sidecar-only derivation evidence | v1.10 contract | Public benchmark artifacts remain stable. [VERIFIED: docs/internal/solar_derivation_contract.md] |

**Deprecated/outdated:** Adding SOLAR derivation fields directly to trace JSONL, public Pydantic benchmark schemas, or primary CLI output is out of scope for Phase 48. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The new module should likely be named `solar_derivation.py`. | Recommended Implementation Strategy | Low; planner can choose a different internal name. |
| A2 | The top-level artifact class could be called `SolarDerivationEvidence`. | Recommended Implementation Strategy | Low; class name is discretionary. |
| A3 | Minimal family recognizers can use fixture-driven heuristics before full formulas. | Recommended Implementation Strategy | Medium; planner must avoid overclaiming supported behavior. |
| A4 | New tests should be placed in `tests/sol_execbench/test_solar_derivation_evidence.py`. | Validation Architecture | Low; file name is discretionary. |
| A5 | The schema version should be separate from AMD SOL v2, for example `sol_execbench.solar_derivation.v1`. | Recommended Implementation Strategy | Low; exact string is discretionary but must be unique. |

## Open Questions

1. **Should Phase 48 expose package-level imports from `sol_execbench.core.scoring`?**
   - What we know: Existing v2 artifacts and bound graph APIs are exported from `__init__.py`. [VERIFIED: src/sol_execbench/core/scoring/__init__.py]
   - What's unclear: Phase context says export only if later internal callers need package-level imports. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md]
   - Recommendation: Keep imports module-local in Phase 48 unless implementation tests need the public scoring package path. [ASSUMED]

2. **How much fixture execution should Phase 48 attempt?**
   - What we know: Fixture JSON files contain reference snippets and workload axes, but not complete public `Definition` input/output schemas. [VERIFIED: tests/sol_execbench/fixtures/solar_derivation/attention_positive.json]
   - What's unclear: Whether plans should synthesize `Definition` objects for every fixture in Phase 48. [ASSUMED]
   - Recommendation: Validate all fixtures at the contract level and run extractor integration on representative hand-authored definitions for each confidence state. [ASSUMED]

3. **Should semantic axes be inferred from axis names, tensor dimension positions, or both?**
   - What we know: `Definition.axes`, `TensorSpec.shape`, and `Workload.axes` expose named axes and resolved concrete values. [VERIFIED: src/sol_execbench/core/data/definition.py] [VERIFIED: src/sol_execbench/core/data/workload.py]
   - What's unclear: The exact semantic-axis normalization vocabulary beyond fixture-required evidence strings. [ASSUMED]
   - Recommendation: Preserve raw axis names plus normalized fixture evidence strings, and avoid broad semantic renaming until later family formula phases. [ASSUMED]

## Sources

### Primary (HIGH confidence)

- `.planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md` - locked decisions, boundaries, confidence rules, and test contract.
- `.planning/REQUIREMENTS.md` - DERIVE-07, MODEL-03, MODEL-04, and out-of-scope constraints.
- `.planning/ROADMAP.md` - Phase 48 goal, dependencies, success criteria, and v1.10 milestone boundaries.
- `.planning/STATE.md` - current project state and deferred items.
- `AGENTS.md` - repository structure, coding style, test commands, and security constraints.
- `docs/internal/solar_derivation_contract.md` - Phase 47 sidecar-only derivation contract, fixture schema, states, and warning prefixes.
- `tests/sol_execbench/solar_derivation_fixtures.py` - fixture loader and validation vocabulary.
- `src/sol_execbench/core/scoring/amd_bound_graph.py` - current FX/AST extraction and bound graph model.
- `src/sol_execbench/core/scoring/amd_bound_estimates.py` - current formula/byte estimate model.
- `src/sol_execbench/core/scoring/amd_sol_v2.py` - sidecar artifact, parser, aggregate status, warnings, and coverage patterns.
- `tests/sol_execbench/test_amd_bound_graph.py` - graph extraction and metadata behavior.
- `tests/sol_execbench/test_amd_sol_v2.py` - sidecar round trip, confidence, warnings, and v1/v2 boundary tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` - public schema, CLI, and claim guardrails.

### Secondary (MEDIUM confidence)

- None. Research used local authoritative project sources only.

### Tertiary (LOW confidence)

- None. Assumptions are explicitly listed in the assumptions log.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - local package config and tool versions were verified. [VERIFIED: pyproject.toml] [VERIFIED: uv run python --version] [VERIFIED: uv run pytest --version]
- Architecture: HIGH - current scoring modules and tests directly show the integration path. [VERIFIED: src/sol_execbench/core/scoring/amd_bound_graph.py] [VERIFIED: src/sol_execbench/core/scoring/amd_sol_v2.py]
- Pitfalls: HIGH - most risks are locked by Phase 48 context and existing guardrails. [VERIFIED: .planning/phases/48-extraction-pipeline-and-semantic-provenance/48-CONTEXT.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- Exact grouping heuristics: MEDIUM - the fixture contract is verified, but full family recognizers are intentionally deferred. [VERIFIED: docs/internal/solar_derivation_contract.md] [VERIFIED: .planning/ROADMAP.md]

**Research date:** 2026-05-23  
**Valid until:** 2026-06-22, unless Phase 49 or Phase 51 changes the sidecar contract earlier.

## RESEARCH COMPLETE
