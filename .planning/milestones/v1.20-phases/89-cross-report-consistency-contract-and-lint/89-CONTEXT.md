# Phase 89: Cross-Report Consistency Contract And Lint - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers a deterministic, CPU-safe cross-report consistency contract
and lint surface over existing v1.19-era evidence reports: execution closure,
paper denominator, ROCm Compatibility Matrix, runtime evidence, static evidence,
AMD score, AMD SOL/SOLAR, and AMD bound sanity. It detects contradictory
status, provenance, checksum, and claim-boundary combinations without changing
canonical Trace, score, timing, correctness, evaluator, or public schema
semantics.

</domain>

<decisions>
## Implementation Decisions

### Scope And Inputs
- Treat consistency lint as a new sidecar/reporting layer over already-produced
  JSON reports, not as a replacement for any existing report builder.
- Load report payloads from explicit caller-supplied paths or source refs; do
  not discover arbitrary files recursively or infer authority from directory
  layout.
- Support the current evidence surface first: execution closure, paper
  denominator, ROCm Compatibility Matrix, runtime evidence, static evidence,
  AMD score, AMD SOL/SOLAR, and AMD bound sanity.
- Missing optional inputs should be classified as missing evidence or
  informational context unless a requested contradiction check requires them.

### Contradiction Model
- Start with high-value contradictions mapped directly to requirements:
  attempted/blocked denominator drift, Matrix runtime-unavailable versus
  attempted execution evidence, missing-derived-evidence versus scored reports,
  stale or mismatched refs/checksums, and claim-boundary violations.
- Use stable severities such as blocker, warning, and info, with reason codes
  that are explicit enough for tests and downstream tooling.
- Prefer deterministic rule helpers over generic deep-diff logic; each finding
  should have a specific source pair, reason code, severity, and remediation
  hint.
- Keep false authority boundaries literal and visible. A consistency pass can
  mean "internally coherent"; it must not mean "validated" or "authoritative."

### Output Shape
- Emit strict Pydantic models plus deterministic JSON and Markdown renderers,
  following existing patterns in paper denominator, Matrix diff, and AMD bound
  sanity reports.
- Use bounded relative refs and checksums instead of embedding full payloads,
  raw logs, proprietary kernels, credentials, or absolute temporary paths.
- Include a concise aggregate summary plus per-finding details so the artifact
  works for both human review and machine gatekeeping.
- Expose generation through a thin script, not a primary `sol-execbench` CLI
  option, unless a later phase explicitly promotes it.

### Testing And Contracts
- Cover model validation, deterministic serialization, finding classification,
  Markdown rendering, bounded refs, and diagnostic-only guardrails with
  CPU-safe tests.
- Use small synthetic report fixtures rather than real hardware runs for the
  core contract.
- Add public contract guardrails proving canonical Trace, Definition, Workload,
  Solution, correctness, timing, score, and evaluator semantics remain
  unchanged.
- Avoid dependency relocking, Docker privilege changes, database additions, or
  remote service assumptions.

### the agent's Discretion
- The agent may choose exact module names, enum names, and helper boundaries as
  long as they follow the existing `core/*` sidecar/reporting patterns.
- The agent may split implementation into one or more plans depending on test
  blast radius, but Phase 89 should stay focused on consistency lint only and
  not implement evaluation stability or trust summary logic early.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sol_execbench/core/dataset/execution_closure.py` defines the execution
  closure sidecar contract and provenance helpers.
- `src/sol_execbench/core/dataset/paper_denominator.py` defines denominator
  reporting patterns, deterministic JSON, Markdown rendering, refs, and
  claim-boundary wording.
- `src/sol_execbench/core/compatibility.py` and
  `src/sol_execbench/core/matrix_diff.py` define ROCm Matrix report and diff
  patterns that can inform severity and source-pair handling.
- `src/sol_execbench/core/runtime_evidence.py`,
  `src/sol_execbench/core/bench/static_kernel_evidence.py`, and
  `src/sol_execbench/core/scoring/*` provide the evidence report families this
  phase needs to reference.
- `src/sol_execbench/core/reporting.py` and dataset checksum helpers are likely
  reusable for bounded refs, deterministic JSON, and Markdown output.

### Established Patterns
- Sidecar/report models use strict Pydantic-style schemas, explicit
  `schema_version` values, deterministic serialization, and CPU-safe tests.
- Diagnostic evidence is exposed through thin scripts when it is a research
  artifact rather than primary benchmark execution behavior.
- Public contract guardrails assert that optional evidence capabilities do not
  mutate canonical trace, timing, correctness, score, or evaluator fields.
- Tests prefer synthetic fixtures and direct model assertions for sidecar
  contracts, with focused script/parser coverage where a script is added.

### Integration Points
- A likely core module is `src/sol_execbench/core/consistency.py` or a similarly
  scoped reporting module.
- A likely script is `scripts/report_consistency.py` or
  `scripts/lint_evidence_consistency.py`, accepting explicit JSON paths and
  optional JSON/Markdown output paths.
- Tests should live under `tests/sol_execbench/test_consistency_report.py` and
  `tests/sol_execbench/test_consistency_script.py`, with any public contract
  guardrails added to existing guardrail tests when appropriate.
- Future phases should consume Phase 89 output rather than reimplementing
  contradiction rules.

</code_context>

<specifics>
## Specific Ideas

- Keep this phase script-side and diagnostic-only.
- Prefer explicit source refs and checksums over payload duplication.
- Build contradiction rules that future claim-upgrade and trust-summary phases
  can consume.
- Keep v1.20 no-new-hardware and no-authority-upgrade boundaries visible in
  both JSON and Markdown output.

</specifics>

<deferred>
## Deferred Ideas

- `evaluation_stability.v1` belongs to Phase 90.
- Claim-upgrade rules belong to Phase 91.
- Trust summary integration belongs to Phase 92.
- Public docs/examples across the whole v1.20 surface belong to Phase 93.
- MI300X-on-CDNA3 and CDNA4/native-host validation, full paper parity, upstream SOLAR
  parity, and hosted leaderboard readiness remain outside v1.20.

</deferred>
