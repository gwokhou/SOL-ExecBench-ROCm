# Roadmap: SOL ExecBench ROCm Port

## Current Milestone: v1.28 CDNA3 Test and Documentation Readiness

**Goal:** Complete the CDNA3/gfx94* testing, documentation, and evidence
readiness surface so future MI300X validation has concrete commands, test entry
points, artifact requirements, and claim guardrails, while actual hardware
execution remains deferred on the current machine.

**Scope boundary:** This milestone prepares CDNA3 tests and documentation for
real hardware use. It does not claim CDNA3 hardware validation, run MI300X on
the current host, upgrade score/report authority, validate CDNA4, or perform
paper-scale validation.

## Phases

### Phase 127: CDNA3 Hardware-Gated Test Surface

**Status:** Pending

**Goal:** Add concrete CDNA3-only pytest coverage and selection semantics so
`requires_cdna3` is an actionable hardware-gated test path rather than only a
registered marker.

**Requirements:** CDNA3-TEST-01, CDNA3-TEST-02, CDNA3-TEST-03, CDNA3-TEST-04

**Success criteria:**
1. At least one test uses `@pytest.mark.requires_cdna3` and runs only when the
   detected ROCm architecture is `gfx94*`.
2. Non-CDNA3 hosts skip CDNA3 tests with clear messages that identify the
   detected architecture or missing ROCm state.
3. CPU-safe tests validate marker registration, architecture detection, and
   CDNA3 metadata handling without requiring MI300X access.
4. Test-suite audit coverage fails if the repository regresses to a marker-only
   CDNA3 surface with no concrete hardware-gated test.

### Phase 128: MI300X Evidence Contract and Validation Handoff

**Status:** Pending

**Goal:** Turn MI300X/gfx942 validation readiness into a concrete future-run
contract covering commands, artifacts, environment capture, timing evidence,
clock policy, traces, score reports, and expected failure/skip handling.

**Requirements:** MI300X-EVID-01, MI300X-EVID-02, MI300X-EVID-03,
MI300X-EVID-04

**Success criteria:**
1. Handoff docs provide an ordered MI300X validation command sequence for
   environment discovery, full pytest, dataset execution, timing evidence,
   trace capture, and score reporting.
2. Required artifacts are named and tied to claim-upgrade gates before any
   support matrix can mark CDNA3/MI300X as hardware-validated.
3. The validation contract distinguishes expected skips, missing tools,
   functional failures, timing instability, missing evidence, and deferred
   quantization formats.
4. FP8 readiness is documented for MI300X when FP8 workloads are present, while
   NVFP4/MXFP4 remains explicitly deferred.

### Phase 129: Deferred-Execution Guardrails

**Status:** Pending

**Goal:** Strengthen CPU-safe guardrails so CDNA3 schema/build/test readiness
cannot be mistaken for completed hardware validation.

**Requirements:** CDNA3-GATE-01, CDNA3-GATE-02, CDNA3-GATE-03, CDNA3-GATE-04

**Success criteria:**
1. Existing and new CPU-safe tests fail if docs or reports claim CDNA3 hardware
   validation without complete real-hardware evidence.
2. Readiness and claim-upgrade helpers continue to surface blockers for missing
   MI300X identity, `gfx942` architecture, full-suite result, clocks, traces,
   timing evidence, and score artifacts.
3. Score/report warning text remains present for `gfx94*` artifacts until the
   future evidence chain removes blockers.
4. Public release and research-preview materials retain deferred CDNA3 wording
   after the new hardware-gated tests are added.

### Phase 130: CDNA3 Public Documentation Closure

**Status:** Pending

**Goal:** Update public and contributor-facing documentation so users know how
to run CDNA3 tests, interpret skips, add future CDNA3 coverage, and understand
the difference between readiness and validation.

**Requirements:** CDNA3-DOC-01, CDNA3-DOC-02, CDNA3-DOC-03, CDNA3-DOC-04

**Success criteria:**
1. `docs/TESTING.md` documents CDNA3-only pytest commands, expected skips on
   non-CDNA3 hosts, and current-machine execution limits.
2. ROCm support docs clearly separate CDNA3 schema support, CDNA3 test
   readiness, deferred MI300X full-suite validation, and unavailable CDNA4.
3. Internal handoff docs name the exact future evidence gate for upgrading
   CDNA3 support status.
4. Contributor docs explain where future CDNA3 tests belong and which markers,
   evidence artifacts, and claim boundaries apply.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CDNA3-TEST-01 | Phase 127 | Pending |
| CDNA3-TEST-02 | Phase 127 | Pending |
| CDNA3-TEST-03 | Phase 127 | Pending |
| CDNA3-TEST-04 | Phase 127 | Pending |
| MI300X-EVID-01 | Phase 128 | Pending |
| MI300X-EVID-02 | Phase 128 | Pending |
| MI300X-EVID-03 | Phase 128 | Pending |
| MI300X-EVID-04 | Phase 128 | Pending |
| CDNA3-GATE-01 | Phase 129 | Pending |
| CDNA3-GATE-02 | Phase 129 | Pending |
| CDNA3-GATE-03 | Phase 129 | Pending |
| CDNA3-GATE-04 | Phase 129 | Pending |
| CDNA3-DOC-01 | Phase 130 | Pending |
| CDNA3-DOC-02 | Phase 130 | Pending |
| CDNA3-DOC-03 | Phase 130 | Pending |
| CDNA3-DOC-04 | Phase 130 | Pending |

**Coverage:** 16/16 v1.28 requirements mapped.
**Progress:** 0/16 requirements complete; 0/4 phases complete.

## Completed Milestones

- Complete **v1.27 Copyright Provenance Cleanup** - Phases 123-126
  (shipped 2026-06-02). See `.planning/milestones/v1.27-ROADMAP.md`.
- Complete **v1.26 Public Prerelease and Research Preview** - Phases 119-122
  (shipped 2026-06-02). See `.planning/milestones/v1.26-ROADMAP.md`.
- Complete **v1.25 Engineering Prerelease** - Phases 114-118
  (shipped 2026-06-01). See `.planning/milestones/v1.25-ROADMAP.md`.
- Complete **v1.24 Dataset Batch Run Trustworthiness** - Phases 110-113
  (shipped 2026-06-01). See `.planning/milestones/v1.24-ROADMAP.md`.
- Complete **v1.23 Evaluation Reliability and Security Hardening** -
  Phases 106-109 (shipped 2026-06-01). See
  `.planning/milestones/v1.23-ROADMAP.md`.
- Earlier milestones are archived under `.planning/milestones/`.

## Archived Phase Index

### Phase 123: Provenance Classification Policy

**Status:** Complete. Archived in
`.planning/milestones/v1.27-phases/123-provenance-classification-policy/`.

### Phase 124: SPDX Header Cleanup

**Status:** Complete. Archived in
`.planning/milestones/v1.27-phases/124-spdx-header-cleanup/`.

### Phase 125: Compliance And Attribution Documentation

**Status:** Complete. Archived in
`.planning/milestones/v1.27-phases/125-compliance-and-attribution-documentation/`.

### Phase 126: Provenance Guardrails And Release Gates

**Status:** Complete. Archived in
`.planning/milestones/v1.27-phases/126-provenance-guardrails-and-release-gates/`.

## Current Position

**Status:** v1.28 requirements and roadmap defined. Ready to discuss or plan
Phase 127.

Run `$gsd-discuss-phase 127` to gather implementation context, or
`$gsd-plan-phase 127` to plan directly.
