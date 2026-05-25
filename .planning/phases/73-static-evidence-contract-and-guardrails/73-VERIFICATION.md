---
phase: 73-static-evidence-contract-and-guardrails
verified: 2026-05-25T16:35:22Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Phase 73: Static Evidence Contract And Guardrails Verification Report

**Phase Goal:** Maintainers and consumers have a strict diagnostic-only static evidence contract before artifact collection or reporting exists.
**Verified:** 2026-05-25T16:35:22Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Maintainer can serialize and parse a strict `sol_execbench.static_kernel_evidence.v1` sidecar with stable status, reason-code, artifact, tool-run, and classification fields. | VERIFIED | `src/sol_execbench/core/bench/static_kernel_evidence.py` defines schema version, strict frozen model config, status/reason enums, artifact/tool-run/classification models, and `to_dict()`/`model_dump(mode="json")` support. Tests round-trip `StaticKernelEvidenceSidecar.model_validate(payload)` and reject extra top-level/nested fields. |
| 2 | Consumer can inspect authority metadata showing static evidence is diagnostic only and is not correctness, performance, timing, score, paper-parity, or leaderboard authority. | VERIFIED | `StaticKernelEvidenceSidecar` encodes `diagnostic_only: Literal[True]` and false literal authority fields for correctness, performance, timing, score, paper parity, and leaderboard claims. Tests assert the serialized values. |
| 3 | Maintainer can represent aggregate and per-artifact collected, partial, unavailable, unsupported, failed, and skipped states. | VERIFIED | `StaticKernelEvidenceStatus` locks all six states. Generic sidecar construction supports collected; helper constructors cover partial, unavailable, unsupported, failed, and skipped; artifact entries include per-artifact `status` and `reason_code`. |
| 4 | Consumer can discover optional static evidence support from evaluator contract metadata without a required evaluator contract version bump. | VERIFIED | `build_evaluator_contract()` includes `static_kernel_evidence.v1` in `capabilities` while `SOL_EXECBENCH_CONTRACT_VERSION` and emitted `contract_version` remain `"1.0"`. Contract tests also verify trace/correctness/timing/scoring fields are unchanged and CLI JSON equals the builder payload. |
| 5 | Maintainer can verify static evidence helpers leave canonical trace JSONL, correctness, timing, scoring, and default benchmark behavior unchanged. | VERIFIED | Public guardrail tests exclude static evidence keys from canonical trace payloads and primary CLI help. Reporting/scoring guardrail tests construct a sidecar, summarize traces, score, and compare trace dumps before/after. Grep confirms no static evidence integration in CLI, reporting, scoring, or scoring guardrails. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/bench/static_kernel_evidence.py` | Strict sidecar schema, status/reason vocabulary, authority metadata, pure helpers | VERIFIED | Exists and substantive. SDK artifact verifier passed; manual review found strict config at lines 17-23, enums at 26-52, nested models at 77-186, sidecar at 189-235, helpers at 238-325. |
| `src/sol_execbench/core/data/contract.py` | Optional evaluator contract capability token | VERIFIED | `static_kernel_evidence.v1` appears in `capabilities`; contract version remains `1.0`; static evidence boundary claim is diagnostic-only. |
| `tests/sol_execbench/test_static_kernel_evidence.py` | Schema, strict validation, authority, status/reason, helper tests | VERIFIED | Imports contract models/helpers, verifies round-trip, extra-field rejection, locked vocabularies, authority booleans, helper sidecars, and per-artifact classification. |
| `tests/sol_execbench/test_contract.py` | Evaluator capability and version guardrails | VERIFIED | Verifies optional capability, unchanged contract version, unchanged trace/correctness/timing/scoring field groups, and CLI JSON builder equivalence. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Canonical trace and primary CLI negative guardrails | VERIFIED | Excludes static evidence key space from canonical trace JSON and excludes Phase 76 `--static-evidence` options from primary CLI help. |
| `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` | Trace/report/scoring isolation tests | VERIFIED | Imports sidecar helper only inside guardrail scope and verifies sidecar construction does not mutate trace dumps or scoring behavior. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/sol_execbench/test_static_kernel_evidence.py` | `src/sol_execbench/core/bench/static_kernel_evidence.py` | Imports strict Pydantic models and pure constructors | WIRED | Imports at test lines 6-20 exercise models/helpers throughout the file. |
| `src/sol_execbench/core/data/contract.py` | `tests/sol_execbench/test_contract.py` | `build_evaluator_contract()` capability list | WIRED | Capability is emitted at contract line 84; tests assert optional capability/version/field boundaries at lines 48-66 and CLI JSON at lines 118-127. SDK pattern check missed this because the plan regex was escaped. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | `sol_execbench.cli.main:cli` | `CliRunner --help` assertion | WIRED | Test invokes `cli` and excludes static evidence CLI flags at lines 263-293. |
| `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` | `src/sol_execbench/core/data/trace.py` | Trace model dump before/after isolation checks | WIRED | Test imports `Trace` and related trace models at lines 8-15 and compares `model_dump(mode="json")` before/after at lines 131-150. SDK pattern check missed this because it looked for a direct target filename reference. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `StaticKernelEvidenceSidecar` | `status`, `reason_code`, `artifacts`, `tool_runs`, `classification`, authority fields | Constructor/helper arguments plus strict Pydantic validation | Yes | VERIFIED |
| `EvaluatorContract.capabilities` | `capabilities` list | `build_evaluator_contract()` static metadata | Yes | VERIFIED |
| Guardrail tests | Trace dumps, CLI help text, scoring output | Existing trace/CLI/reporting/scoring APIs invoked in tests | Yes | VERIFIED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 73 focused tests pass | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q` | `53 passed in 2.97s` | PASS |
| Phase 73 files pass Ruff | `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/static_kernel_evidence.py src/sol_execbench/core/data/contract.py tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` | `All checks passed!` | PASS |
| Sidecar serialize/parse spot-check | `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "...build_static_kernel_evidence_skipped..."` | Printed `sol_execbench.static_kernel_evidence.v1 skipped True False` and parsed status `skipped` | PASS |
| No discovery, subprocess, or file-writing behavior in contract module | `rg -n "subprocess|shutil\\.which|rglob|glob|write_text|open\\(" src/sol_execbench/core/bench/static_kernel_evidence.py` | No matches | PASS |
| No premature runtime integration | `rg -n "static-evidence|static_evidence|StaticKernelEvidence" src/sol_execbench/cli src/sol_execbench/core/reporting.py src/sol_execbench/sol_score.py src/sol_execbench/core/scoring_guardrails.py` | No matches | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| Conventional phase probes | `find scripts -path '*/tests/probe-*.sh' -type f` and phase plan/summary probe grep | No probes found or declared for Phase 73 | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SKE-CONTRACT-01 | `73-01-PLAN.md` | Strict sidecar schema with stable status, reason-code, artifact, tool-run, and classification fields | SATISFIED | Static module models and tests cover schema version, round-trip parsing, strict extra-field rejection, vocabularies, artifact/tool-run/classification fields. |
| SKE-CONTRACT-02 | `73-01-PLAN.md` | Diagnostic-only authority boundaries with explicit false authority fields | SATISFIED | Sidecar literal authority fields and authority serialization test. |
| SKE-CONTRACT-03 | `73-01-PLAN.md` | Aggregate and per-artifact states for collected, partial, unavailable, unsupported, failed, skipped | SATISFIED | Status enum, non-collected helper tests, representative collected sidecar, and per-artifact status/reason assertions. |
| SKE-CONTRACT-04 | `73-01-PLAN.md` | Optional evaluator capability without required version bump | SATISFIED | Capability token in contract metadata; version remains `1.0`; CLI contract JSON builder equivalence test. |
| SKE-CONTRACT-05 | `73-01-PLAN.md` | Static evidence helpers do not mutate trace JSONL, correctness, timing, scoring, default behavior | SATISFIED | Canonical trace/CLI exclusion tests, no-mutation report/scoring tests, no runtime integration grep. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| Phase files | n/a | Debt markers `TBD`, `FIXME`, `XXX`, placeholder implementation, console-only handlers | None | No blocker markers or implementation stubs found. Empty-list matches are intentional helper/guardrail assertions, not user-visible stub data. |

### Human Verification Required

None. Phase 73 is contract/test-only, and `.planning/phases/73-static-evidence-contract-and-guardrails/73-VALIDATION.md` states all phase behaviors have automated verification with no live ROCm or manual validation required.

### Gaps Summary

No blocking gaps found. The phase goal is achieved: the strict diagnostic-only sidecar contract exists, consumers can discover optional support through evaluator metadata without a contract bump, and automated guardrails verify static evidence remains sidecar-only before artifact collection, CLI integration, reporting, or live validation are introduced in later phases.

---

_Verified: 2026-05-25T16:35:22Z_
_Verifier: the agent (gsd-verifier)_
