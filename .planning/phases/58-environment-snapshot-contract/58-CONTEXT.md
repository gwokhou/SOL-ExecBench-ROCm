# Phase 58: Environment Snapshot Contract - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning
**Source:** v1.13 roadmap and ROCm Systems/GPUOpen enhancement note

<domain>
## Phase Boundary

Phase 58 defines the internal environment evidence contract and collection
boundary for v1.13. It does not attach snapshots to benchmark traces yet; that
belongs to Phase 59. It does not add profiler artifacts; that belongs to v1.14.

The deliverable is a reusable, testable core layer that can represent ROCm/GPU
environment evidence, run bounded tool probes, and advertise optional evidence
support through the evaluator contract while keeping contract version `1.0`.
</domain>

<decisions>
## Locked Decisions

### Compatibility
- Keep `sol_execbench.evaluator_contract.v1` and contract version `1.0`.
- Add optional capability token `runtime.evidence.v1`; do not require consumers
  to understand it.
- Do not mutate `Trace`, `Evaluation`, `Environment`, `Definition`,
  `Workload`, or `Solution` public schemas in Phase 58.
- Missing ROCm tools must produce structured unavailable/error statuses, not
  exceptions that fail normal imports or contract generation.

### Implementation Shape
- Put the reusable snapshot contract in the core layer, close to existing
  diagnostics and trace/environment concepts.
- Use Pydantic models for serialized evidence so downstream result metadata can
  consume stable JSON later.
- Keep subprocess probing injectable so unit tests can simulate `amd-smi`,
  `rocminfo`, and `rocm_agent_enumerator` without requiring ROCm hardware.
- Bound every external tool probe with timeouts and captured stdout/stderr
  tails.

### Deferred
- Benchmark run integration is Phase 59.
- `sol-execbench doctor` / `env-snapshot` CLI and smoke checks are Phase 60.
- `rocprofv3` artifact lifecycle is v1.14.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before implementing Phase 58.**

### Planning
- `.planning/milestones/v1.13-REQUIREMENTS.md` — active v1.13 requirements.
- `.planning/milestones/v1.13-ROADMAP.md` — Phase 58-60 structure.
- `.planning/notes/2026-05-23-rocm-systems-gpuopen-enhancements.md` — source
  research note.

### Existing Contracts
- `src/sol_execbench/core/data/contract.py` — evaluator contract builder and
  capability list.
- `src/sol_execbench/core/data/trace.py` — existing canonical trace
  environment schema that must not be expanded in Phase 58.
- `tests/sol_execbench/test_contract.py` — contract tests to extend.
- `tests/sol_execbench/test_public_contract_guardrails.py` — public schema and
  trace guardrails.

### Existing Diagnostics
- `src/sol_execbench/core/diagnostics.py` — current diagnostic helper patterns,
  tool detection, gfx classification, and library readiness structures.
- `tests/conftest.py` — existing ROCm/RDNA4/CDNA3 marker detection behavior.
</canonical_refs>

<specifics>
## Specific Ideas

- Snapshot status vocabulary should be explicit and low-cardinality:
  `available`, `unavailable`, `failed`, `timeout`, `skipped`.
- Tool evidence should include command, detected path when known, status,
  return code, stdout/stderr tails, timeout seconds, and parsed fields.
- Snapshot top-level metadata should include schema version, generated time,
  tools dictionary, optional GPU summaries, optional PyTorch ROCm summary, and
  warnings.
- Tool parsers should be conservative; raw output tails are enough for Phase 58
  where robust parsing is not yet required for benchmark behavior.
</specifics>

<deferred>
## Deferred Ideas

- Do not make environment evidence mandatory for hip-playground
  `confirmed_benchmark` policy in this repository.
- Do not implement RGA/code-object/ISA analysis in v1.13.
- Do not add live GPU memory copy/event timing in Phase 58; preflight smoke
  checks belong to Phase 60.
</deferred>

---

*Phase: 58-environment-snapshot-contract*
*Context gathered: 2026-05-25*

