# Research: Stack for v1.36 SOL Agent Feedback Sidecar Producer

**Date:** 2026-06-15
**Scope:** Local project research for SOL-side support needed by HIP Playground
v1.26.

## Existing SOL Stack

- Python 3.12 package under `src/sol_execbench/`.
- Pydantic v2 schemas with strict/frozen model patterns in `core/data/` and
  diagnostic sidecar modules.
- Click/Rich CLI in `src/sol_execbench/cli/main.py`.
- Existing GPU-free compatibility contract in
  `src/sol_execbench/core/data/contract.py`.
- Existing diagnostic-only sidecar patterns:
  - `sol_execbench.static_kernel_evidence.v1`
  - `sol_execbench.rocprofv3_profile.v1`
  - `sol_execbench.rocprofv3_timing.v1`
  - `sol_execbench.evaluation_stability.v1`
  - no-trace diagnostics and environment snapshot sidecars

## Needed Additions

- Add SOL-owned agent-feedback/profile-summary schemas, most likely under
  `src/sol_execbench/core/bench/` or `src/sol_execbench/core/data/` depending
  on whether the sidecar is generated from one evaluation run or exported as a
  cross-run contract.
- Extend `build_evaluator_contract()` with optional capability tokens without
  changing canonical trace field groups or required compatibility fields.
- Reuse existing checksum helpers from `core.dataset.checksums` and evidence
  reference helpers from `core.dataset.evidence_refs` for stable identity and
  compact artifact citations.
- Add tests beside existing contract, static-evidence, profiler, trace, and
  evaluation-stability coverage.

## Non-Changes

- No new runtime dependency is required.
- No canonical Trace JSONL schema change should be made.
- No benchmark scoring, correctness, timing, claim-upgrade, release-gate, or
  cutover semantics should consume this sidecar as authority.
