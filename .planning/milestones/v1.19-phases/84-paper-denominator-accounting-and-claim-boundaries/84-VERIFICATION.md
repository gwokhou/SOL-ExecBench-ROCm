---
phase: 84-paper-denominator-accounting-and-claim-boundaries
verified: 2026-05-31T08:56:18Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: passed
  previous_score: 7/7
  gaps_closed: []
  gaps_remaining: []
  regressions: []
  review_fix_commits:
    - 6de979f
    - 8b741df
---

# Phase 84: Paper Denominator Accounting And Claim Boundaries Verification Report

**Phase Goal:** Researchers can produce a deterministic paper denominator report that accounts for ready, blocked, unsupported, deferred, attempted, filtered, skipped, and evidence-missing benchmark status without presenting the accounting as paper validation.
**Verified:** 2026-05-31T08:56:18Z
**Status:** passed
**Re-verification:** Yes - focused re-verification after review-fix commits `6de979f` and `8b741df`

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Researcher can generate `paper_denominator_report.v1` JSON that rolls up public benchmark denominator status by problem, workload, category, readiness, closure status, and evidence gap. | VERIFIED | `build_paper_denominator_report()` builds strict report models with suite/category/problem/workload rollups. Review fix `8b741df` preserves inventory-only denominator records as `not_attempted`; `test_paper_denominator_report_keeps_inventory_only_denominator_records` verifies one inventory-only problem with two workloads remains in suite/category/problem/workload output. |
| 2 | Researcher can inspect stable denominator reason codes that separate ready, blocked, unsupported, deferred, evidence-missing, attempted-passed, attempted-failed, filtered, skipped, and not-attempted states. | VERIFIED | `DENOMINATOR_STATE_KEYS`, `_readiness_state()`, and `_closure_state()` define the fixed vocabulary; tests verify the full state set and reason buckets. The inventory-only test verifies default `not_attempted` accounting when a workload has no readiness or closure record. |
| 3 | Report consumers can trace denominator conclusions back to source manifest, inventory, readiness, ready-subset, closure, AMD score, AMD SOL, and SOLAR artifacts by path/ref and checksum. | VERIFIED | `PaperDenominatorSources`, `_source()`, and `_artifact_source()` preserve bounded `path`, `ref`, `schema_version`, and `checksum` refs without embedding source payload arrays. Script tests verify artifact path/checksum refs flow from local files. |
| 4 | Researcher can read deterministic Markdown counts, evidence gaps, deferred buckets, and next-evidence hints with paper parity, upstream SOLAR parity, leaderboard authority, native-host validation, and new-hardware validation explicitly kept false. | VERIFIED | `render_paper_denominator_markdown()` emits deterministic sections for status buckets, category counts, evidence gaps, deferred buckets, next evidence hints, sources, and claim boundaries. Review fix `6de979f` added `_md_cell()` escaping for dynamic Markdown table cells; `test_paper_denominator_markdown_escapes_dynamic_table_cells` covers pipes and newlines in artifact refs. |
| 5 | Researcher can run a thin script to write paper denominator JSON and Markdown from existing sidecar files. | VERIFIED | `scripts/report_paper_denominator.py` is an argparse wrapper over `load_json`, `build_paper_denominator_report`, and `write_paper_denominator_reports`; script tests write deterministic JSON and Markdown from fixture sidecars. |
| 6 | Public canonical Definition, Workload, Trace, scoring, timing, and primary CLI contracts do not expose paper denominator fields. | VERIFIED | Guardrail tests assert canonical Definition/Workload/Trace payloads lack paper denominator schema, state, source-ref, and claim-boundary fields; primary `sol-execbench --help` excludes paper-denominator script-only options. |
| 7 | Script output preserves bounded refs/checksums and explicit false claim boundaries. | VERIFIED | Script test verifies JSON sources for AMD SOL and SOLAR artifact paths and false `paper_parity`; core source-ref tests verify source payload internals such as `workloads`, `records`, and `scores` are not embedded. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/sol_execbench/core/dataset/paper_denominator.py` | Strict sidecar models, builder, checksum, JSON serialization, Markdown renderer, write helpers | VERIFIED | Exists and substantive. `gsd-sdk query verify.artifacts` passed; source contains strict `ConfigDict(extra="forbid")` models, stable checksum via `stable_json_checksum`, deterministic `to_json()`, optional-source evidence-missing handling, inventory-only preservation, and Markdown escaping. |
| `tests/sol_execbench/test_paper_denominator_report.py` | CPU-safe DENOM-01..DENOM-05 report tests plus review-fix regression coverage | VERIFIED | Covers rollups, reason buckets, bounded refs, false claim boundaries, deterministic JSON/Markdown, strict unknown-field rejection, missing optional evidence sources, Markdown escaping, and inventory-only denominator records. |
| `scripts/report_paper_denominator.py` | Thin argparse wrapper over core helpers | VERIFIED | Loads local JSON sidecars, passes bounded artifact paths, delegates build/write to core helpers, and does not duplicate report implementation. |
| `src/sol_execbench/core/dataset/__init__.py` | Dataset helper exports | VERIFIED | Exports `PaperDenominatorReport`, `build_paper_denominator_report`, `render_paper_denominator_markdown`, and `write_paper_denominator_reports`. |
| `tests/sol_execbench/test_paper_denominator_script.py` | CPU-safe script JSON/Markdown output tests | VERIFIED | Imports the script directly, writes fixture inputs, verifies deterministic JSON/Markdown and bounded artifact refs. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Sidecar-only and CLI guardrails | VERIFIED | Contains Phase 84 sidecar-only contract test and primary CLI exclusion test. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `paper_denominator.py` | `checksums.py` | `stable_json_checksum` | WIRED | `PaperDenominatorReport.with_checksum()` computes checksum over normalized report payload. `gsd-sdk query verify.key-links` passed. |
| `paper_denominator.py` | Phase 83 closure vocabulary | `attempted_passed`, `attempted_failed`, `derived_evidence_missing`, etc. | WIRED | `_closure_state()` maps Phase 83 closure statuses into fixed denominator state keys. `gsd-sdk query verify.key-links` passed. |
| `scripts/report_paper_denominator.py` | `paper_denominator.py` | builder, loader, writer imports | WIRED | Script imports and calls core helpers directly. `gsd-sdk query verify.key-links` passed. |
| `test_public_contract_guardrails.py` | canonical public contracts | Definition/Workload/Trace payload checks | WIRED | Test serializes sample canonical payloads and asserts paper denominator fields are absent. `gsd-sdk query verify.key-links` passed. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `paper_denominator.py` | report rollups, reason buckets, evidence gaps, source refs | `inventory`, `readiness`, `execution_closure`, `amd_score_report`, AMD SOL/SOLAR artifact args | Yes | FLOWING - builder iterates source payload records and optional artifact paths; absent optional evidence sources intentionally emit bounded evidence-missing reason/evidence gap rows. |
| `paper_denominator.py` | inventory-only workload records | `inventory["problems"][*]["workloads"]` | Yes | FLOWING - review fix `8b741df` initializes workloads from inventory and later marks records with no readiness/closure state as `not_attempted`. |
| `scripts/report_paper_denominator.py` | generated report | local JSON sidecar files loaded from CLI paths | Yes | FLOWING - script loads files with `load_json()` and passes them into the core builder before writing outputs. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Focused Phase 84 verification suite | `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_paper_denominator_report.py tests/sol_execbench/test_paper_denominator_script.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_primary_cli_does_not_expose_v1_19_paper_denominator_options -q` | `12 passed in 1.37s` | PASS |
| Artifact verification | `gsd-sdk query verify.artifacts` for both Phase 84 plans | Plan 84-01: 2/2 passed; Plan 84-02: 4/4 passed | PASS |
| Key-link verification | `gsd-sdk query verify.key-links` for both Phase 84 plans | Plan 84-01: 2/2 verified; Plan 84-02: 2/2 verified | PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| Conventional probes | `find scripts -path '*/tests/probe-*.sh' -type f` | No probe files found for this phase. | SKIPPED |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DENOM-01 | 84-01, 84-02 | JSON sidecar rolls up denominator by problem, workload, category, readiness, closure, and evidence gap. | SATISFIED | Core builder and report tests verify schema and rollups; inventory-only regression test proves denominator records are retained without readiness/closure inputs. |
| DENOM-02 | 84-01, 84-02 | Stable denominator state and reason-code accounting. | SATISFIED | Fixed state key tuple, readiness/closure mapping, reason buckets, next-evidence tests, optional-source evidence-missing tests, and inventory-only `not_attempted` tests. |
| DENOM-03 | 84-01, 84-02 | Source manifest, inventory, readiness, ready-subset, closure, AMD score, AMD SOL, and SOLAR refs by path/ref/checksum only. | SATISFIED | Source ref models and tests verify bounded metadata and absence of duplicated source payload arrays; optional absent sources remain bounded `None` or empty refs. |
| DENOM-04 | 84-01, 84-02 | Claim-boundary fields/wording keep paper parity, upstream SOLAR parity, leaderboard authority, native-host validation, and new-hardware validation false. | SATISFIED | `PaperDenominatorClaimBoundary` defaults false; Markdown includes explicit negative claim wording; guardrail test confirms sidecar-only boundary names. |
| DENOM-05 | 84-01, 84-02 | Deterministic Markdown summary includes counts, evidence gaps, deferred buckets, next-evidence hints without claiming validation. | SATISFIED | Renderer and tests verify deterministic Markdown sections, trailing newline, no validation claims, and escaped dynamic table cells. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | - | No `TBD`, `FIXME`, `XXX`, `TODO`, placeholder, empty implementation, or console-only implementation found in Phase 84 files. | INFO | No blocker anti-patterns detected. |

### Human Verification Required

None.

### Gaps Summary

No gaps found. The Phase 84 implementation still satisfies the roadmap success criteria and DENOM-01 through DENOM-05 after review-fix commits `6de979f` and `8b741df`. Focused re-verification confirms the added coverage for missing optional evidence sources, Markdown escaping, and inventory-only denominator records, while canonical public contracts and the primary CLI remain unchanged.

---

_Verified: 2026-05-31T08:56:18Z_
_Verifier: the agent (gsd-verifier)_
