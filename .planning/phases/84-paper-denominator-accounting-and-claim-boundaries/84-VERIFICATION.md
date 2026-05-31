---
phase: 84-paper-denominator-accounting-and-claim-boundaries
verified: 2026-05-31T08:38:45Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
---

# Phase 84: Paper Denominator Accounting And Claim Boundaries Verification Report

**Phase Goal:** Researchers can produce a deterministic paper denominator report that accounts for ready, blocked, unsupported, deferred, attempted, filtered, skipped, and evidence-missing benchmark status without presenting the accounting as paper validation.
**Verified:** 2026-05-31T08:38:45Z
**Status:** passed
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Researcher can generate `paper_denominator_report.v1` JSON that rolls up public benchmark denominator status by problem, workload, category, readiness, closure status, and evidence gap. | VERIFIED | `build_paper_denominator_report()` creates strict report models with suite/category/problem/workload rollups and evidence gaps in `src/sol_execbench/core/dataset/paper_denominator.py`; `test_paper_denominator_report_aggregates_required_rollups` verifies schema, category, problem, workload, readiness, closure, and evidence-missing counts. |
| 2 | Researcher can inspect stable denominator reason codes that separate ready, blocked, unsupported, deferred, evidence-missing, attempted-passed, attempted-failed, filtered, skipped, and not-attempted states. | VERIFIED | `DENOMINATOR_STATE_KEYS` defines the fixed state vocabulary; `_readiness_state()` and `_closure_state()` map readiness and Phase 83 closure statuses deterministically; `test_paper_denominator_report_tracks_reasons_and_next_evidence` verifies the full state set and stable reason buckets. |
| 3 | Report consumers can trace denominator conclusions back to source manifest, inventory, readiness, ready-subset, closure, AMD score, AMD SOL, and SOLAR artifacts by path/ref and checksum. | VERIFIED | `PaperDenominatorSources`, `_source()`, and `_artifact_source()` preserve bounded `path`, `ref`, `schema_version`, and `checksum` refs. Tests verify inventory, closure, AMD SOL, and SOLAR refs and assert source payload internals are not embedded. |
| 4 | Researcher can read deterministic Markdown counts, evidence gaps, deferred buckets, and next-evidence hints with paper parity, upstream SOLAR parity, leaderboard authority, native-host validation, and new-hardware validation explicitly kept false. | VERIFIED | `render_paper_denominator_markdown()` emits deterministic sections for status buckets, category counts, evidence gaps, deferred buckets, next evidence hints, sources, and claim boundaries. `PaperDenominatorClaimBoundary` defaults all authority fields false and tests assert the wording and booleans. |
| 5 | Researcher can run a thin script to write paper denominator JSON and Markdown from existing sidecar files. | VERIFIED | `scripts/report_paper_denominator.py` is an argparse wrapper over `load_json`, `build_paper_denominator_report`, and `write_paper_denominator_reports`; script test writes deterministic JSON and Markdown from fixture sidecars. |
| 6 | Public canonical Definition, Workload, Trace, scoring, timing, and primary CLI contracts do not expose paper denominator fields. | VERIFIED | Guardrail tests assert canonical Definition/Workload/Trace payloads lack paper denominator schema, state, source-ref, and claim-boundary fields; primary `sol-execbench --help` excludes paper-denominator script-only options. |
| 7 | Script output preserves bounded refs/checksums and explicit false claim boundaries. | VERIFIED | Script test verifies JSON sources for AMD SOL and SOLAR artifact paths, deterministic output, Markdown evidence sections, and false `paper_parity`; core tests verify source refs avoid embedded `workloads`, `records`, and `scores`. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/dataset/paper_denominator.py` | Strict sidecar models, builder, checksum, JSON serialization, Markdown renderer, write helpers | VERIFIED | Exists, substantive, strict `ConfigDict(extra="forbid")` models, stable checksum via `stable_json_checksum`, deterministic `to_json()`, builder and renderer implemented. |
| `tests/sol_execbench/test_paper_denominator_report.py` | CPU-safe DENOM-01..DENOM-05 report tests | VERIFIED | Covers rollups, reason buckets, bounded refs, false claim boundaries, deterministic JSON/Markdown, write helpers, and strict unknown-field rejection. |
| `scripts/report_paper_denominator.py` | Thin argparse wrapper over core helpers | VERIFIED | Loads local JSON sidecars, passes bounded artifact paths, delegates build/write to core helpers, no duplicate report implementation. |
| `src/sol_execbench/core/dataset/__init__.py` | Dataset helper exports | VERIFIED | Exports `PaperDenominatorReport`, `build_paper_denominator_report`, `render_paper_denominator_markdown`, and `write_paper_denominator_reports`. |
| `tests/sol_execbench/test_paper_denominator_script.py` | CPU-safe script output tests | VERIFIED | Imports script directly, writes fixture inputs, verifies deterministic JSON/Markdown and bounded artifact refs. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Sidecar-only and CLI guardrails | VERIFIED | Contains Phase 84 sidecar-only contract test and primary CLI exclusion test. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `paper_denominator.py` | `checksums.py` | `stable_json_checksum` | WIRED | `with_checksum()` computes checksum over normalized report payload. |
| `paper_denominator.py` | Phase 83 closure vocabulary | `attempted_passed`, `attempted_failed`, `derived_evidence_missing`, etc. | WIRED | `_closure_state()` maps Phase 83 closure statuses into fixed denominator state keys. |
| `scripts/report_paper_denominator.py` | `paper_denominator.py` | builder, loader, writer imports | WIRED | Script imports and calls core helpers directly. |
| `test_public_contract_guardrails.py` | canonical public contracts | Definition/Workload/Trace payload checks | WIRED | Test serializes sample canonical payloads and asserts paper denominator fields are absent. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `paper_denominator.py` | report rollups, reason buckets, evidence gaps, source refs | `inventory`, `readiness`, `execution_closure`, `amd_score_report`, AMD SOL/SOLAR artifact args | Yes | FLOWING - builder iterates source payload records and optional artifact paths; no static empty report return. |
| `scripts/report_paper_denominator.py` | generated report | local JSON sidecar files loaded from CLI paths | Yes | FLOWING - script loads files with `load_json()` and passes them into the core builder before writing outputs. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Focused Phase 84 verification suite | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_paper_denominator_report.py tests/sol_execbench/test_paper_denominator_script.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_primary_cli_does_not_expose_v1_19_paper_denominator_options -q` | `9 passed in 1.38s` | PASS |
| Script exposes expected standalone reporting options | `uv run python scripts/report_paper_denominator.py --help` | Help lists inventory/readiness/execution-closure, optional source refs, JSON/Markdown outputs, and `--created-at`. | PASS |
| Dataset exports are importable | `uv run python -c "from sol_execbench.core.dataset import ..."` | Printed `PaperDenominatorReport True True True`. | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| Conventional probes | `find scripts -path '*/tests/probe-*.sh' -type f` | No probe files found for this phase. | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DENOM-01 | 84-01, 84-02 | JSON sidecar rolls up denominator by problem, workload, category, readiness, closure, and evidence gap. | SATISFIED | Core builder and report tests verify schema and rollups. |
| DENOM-02 | 84-01, 84-02 | Stable denominator state and reason-code accounting. | SATISFIED | Fixed state key tuple, readiness/closure mapping, reason buckets, and next-evidence tests. |
| DENOM-03 | 84-01, 84-02 | Source manifest, inventory, readiness, ready-subset, closure, AMD score, AMD SOL, and SOLAR refs by path/ref/checksum only. | SATISFIED | Source ref models and tests verify bounded metadata and absence of duplicated source payload arrays. |
| DENOM-04 | 84-01, 84-02 | Claim-boundary fields/wording keep paper parity, upstream SOLAR parity, leaderboard authority, native-host validation, and new-hardware validation false. | SATISFIED | `PaperDenominatorClaimBoundary` defaults false; Markdown includes explicit negative claim wording; guardrail test confirms sidecar-only boundary names. |
| DENOM-05 | 84-01, 84-02 | Deterministic Markdown summary includes counts, evidence gaps, deferred buckets, next-evidence hints without claiming validation. | SATISFIED | Renderer and tests verify deterministic Markdown sections, trailing newline, and no validation claims. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | No `TBD`, `FIXME`, `XXX`, `TODO`, placeholder, empty implementation, or console-only implementation found in Phase 84 files. | INFO | No blocker anti-patterns detected. |

### Human Verification Required

None.

### Gaps Summary

No gaps found. The Phase 84 implementation satisfies the roadmap success criteria and DENOM-01 through DENOM-05 with code, tests, script wiring, sidecar-only guardrails, deterministic serialization/rendering, and explicit false claim boundaries.

---

_Verified: 2026-05-31T08:38:45Z_
_Verifier: the agent (gsd-verifier)_
