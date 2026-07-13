# Phase 78: Matrix Contract And Claim Guardrails - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase defines the diagnostic compatibility Matrix Entry contract for
v1.18. It establishes Target/requested semantics, observed evidence separation,
bounded compatibility statuses, reason-code and artifact fields, and claim
flags that prevent Docker container validation from being described as native
host validation. It does not implement Docker matrix execution, uv/PyTorch wheel
selection, runtime probing, or report generation beyond fixture-backed contract
tests needed to validate the schema and classification semantics.

</domain>

<decisions>
## Implementation Decisions

### Contract Shape
- Put the compatibility contract in a new core model module, likely
  `src/sol_execbench/core/compatibility.py`, following existing Pydantic
  sidecar patterns.
- Use `MatrixEntry` as the primary entry object with a required `target` object
  and a separate `observed` evidence object.
- Always separate requested Target values from observed host, container, Python
  dependency, toolchain, and GPU evidence so mismatch classification is
  deterministic.
- Use strict schema semantics from Phase 78: bounded enums and explicit claim
  flags are part of the first contract, not deferred tightening.

### Status And Blocking Strategy
- `mixed_version` means the Target request and observed environment do not
  match policy, including container ROCm, PyTorch ROCm wheel, Triton, or
  toolchain mismatch.
- Illegal `mixed_version` Targets are blocked during preflight before benchmark
  execution by default.
- A mixed-version debug override may continue probes or smoke execution only
  when explicitly enabled; it cannot produce `container_validated`,
  `host_validated`, or any clean compatibility claim.
- Keep the v1.18 bounded status vocabulary to `host_validated`,
  `container_validated`, `mixed_version`, `pytorch_wheel_unavailable`,
  `runtime_unavailable`, and `not_tested`.

### Claim Boundaries And Tests
- Passing Docker entries must be described as "container ROCm user-space
  validated on recorded host driver/devices", not as native host validation.
- `host_validated` is allowed only for direct native-host validation evidence.
  Docker entries cannot emit it simply because host and container versions
  match.
- Phase 78 tests should be CPU-safe and cover schema serialization, status
  classification, claim flags, mixed-version blocking semantics, and docs
  wording guardrails.
- Compatibility sidecars are diagnostic compatibility evidence only. They must
  not change canonical trace JSONL, correctness, timing, scoring, paper-parity,
  or leaderboard authority.

### the agent's Discretion
The agent may choose exact class, enum, helper, and fixture names as long as the
public schema uses Target/Matrix Entry language, preserves the bounded status
vocabulary, and keeps future Docker/runtime phases able to extend observed
evidence without changing benchmark semantics.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Public Pydantic schemas commonly use `BaseModelWithDocstrings` from
  `src/sol_execbench/core/data/base_model.py` so generated JSON schema includes
  field documentation.
- Existing diagnostic sidecars and optional evidence live outside canonical
  trace JSONL, including environment snapshots, toolchain routing, profiling
  evidence, and static kernel evidence.
- CLI-facing optional evidence is treated as nonfatal and report-oriented.
  Phase 78 should preserve that pattern and focus on pure model and
  classification behavior.

### Established Patterns
- Modules use lowercase `snake_case.py`; classes and Pydantic models use
  `PascalCase`; enum members use uppercase names with string values.
- Schema and domain validation should raise typed or clear `ValueError`s and
  use Pydantic v2 validators where field relationships matter.
- Tests live under `tests/sol_execbench/` and should use CPU-safe fixtures for
  schema/classification behavior.
- New benchmark evidence must avoid mutating public trace, correctness, timing,
  and scoring schemas unless the milestone explicitly requires it.

### Integration Points
- `src/sol_execbench/core/environment.py` already records ROCm/PyTorch runtime
  evidence but does not yet model host/container split or compatibility matrix
  status.
- `src/sol_execbench/core/toolchain.py` provides precedent for routed status
  and reason reporting.
- `docs/user/CLAIMS.md` already carries project claim-boundary language and should
  receive wording guardrails for container user-space vs native host validation
  in later phases.
- `tests/sol_execbench/test_public_contract_guardrails.py` and nearby schema
  tests are likely analogs for guarding public wording and authority boundaries.

</code_context>

<specifics>
## Specific Ideas

- Prefer "Target" over "Row" in user-facing docs and schema descriptions.
- Include `benchmark_allowed` or equivalent behavior in classification output
  so illegal `mixed_version` entries can be blocked before benchmark execution
  while preserving an explicit debug override path for later phases.
- Keep claim flags explicit: diagnostic compatibility evidence can be true,
  while score, paper-parity, leaderboard, and native-host authority remain
  false unless direct evidence supports them.

</specifics>

<deferred>
## Deferred Ideas

- Docker image selection, target manifest parsing, and device preflight belong
  to Phase 79.
- uv index/lock strategy and PyTorch ROCm wheel availability checks belong to
  Phase 80.
- Runtime evidence collection, per-target report emission, and aggregate matrix
  output belong to Phase 81.
- Full documentation workflow, CPU-safe docs guardrails, Docker script tests,
  and marker-gated live ROCm validation guidance belong to Phase 82.

</deferred>
