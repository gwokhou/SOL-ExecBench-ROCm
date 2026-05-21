# Roadmap: SOL ExecBench ROCm Port

**Created:** 2026-05-22
**Milestone:** v1.2 Engineering Practice Harvest and Compatibility Guardrails
**Mode:** standard
**Granularity:** standard

## Overview

This roadmap harvests useful engineering practices from
`~/PyCharmMiscProject/hip-playground/hip-execbench` without turning SOL
ExecBench ROCm into that project and without changing public interfaces or
formats. The milestone focuses on internal robustness, operator clarity,
scoring/comparison review, and compatibility guardrails.

The phase numbering continues from the shipped v1.1 roadmap. v1.1 completed
Phases 7 through 9, so this milestone starts at Phase 10.

CDNA 3 real hardware validation remains deferred by user decision. This
milestone may preserve or improve the handoff, but it must not claim a full
`gfx94*` validation pass.

## Phase Status

- [x] **Phase 10: Practice Harvest and Adaptation Map** (completed 2026-05-22)
- [x] **Phase 11: ROCm Diagnostics and Failure Reporting** (completed 2026-05-22)
- [x] **Phase 12: Scoring and Baseline Comparison Review** (completed 2026-05-22)
- [x] **Phase 13: Public Contract Compatibility Guardrails** (completed 2026-05-22)

## Phases

### Phase 10: Practice Harvest and Adaptation Map

**Goal:** Convert the `hip-execbench` inspection into a concrete adaptation map
that identifies which practices are safe, useful, and compatible with SOL
ExecBench ROCm's existing benchmark contracts.

**Requirements:** HARV-01, HARV-02, HARV-03

**Success Criteria:**
1. A maintained planning or documentation artifact maps selected `hip-execbench` patterns to SOL ExecBench ROCm opportunities with accept/reject rationale.
2. Accepted ideas are explicitly scoped to internal implementation, tests, or documentation.
3. Rejected ideas include a reason when they would change public schemas, CLI behavior, trace formats, or benchmark semantics.
4. The adaptation map calls out CDNA 3 hardware validation as deferred, not part of this milestone's validation evidence.

### Phase 11: ROCm Diagnostics and Failure Reporting

**Goal:** Improve ROCm operator clarity by adapting suitable diagnostic routing,
structured error, and reporting ideas while preserving current command and file
contracts.

**Requirements:** DIAG-01, DIAG-02, DIAG-03

**Success Criteria:**
1. Diagnostics can explain ROCm profiling or hardware-introspection readiness more clearly without requiring a new public CLI contract.
2. Internal failure reporting separates parse, packaging, compile, runtime, verification, timing, and environment failures where that distinction improves actionability.
3. Report or analysis helpers make local ROCm benchmark outcomes easier to inspect while keeping trace JSONL schema compatibility.
4. Focused tests cover the improved diagnostics or reporting paths without requiring unavailable hardware for normal CI.

### Phase 12: Scoring and Baseline Comparison Review

**Goal:** Review `hip-execbench` scoring and baseline-comparison practices and
adapt only the guardrails that improve SOL ExecBench ROCm interpretation without
creating unsupported AMD performance claims.

**Requirements:** SCORE-01, SCORE-02, SCORE-03

**Success Criteria:**
1. A comparison note or code-level guardrail explains which `hip-execbench` scoring or baseline ideas are safe to adapt and which are deferred.
2. Any scoring/reporting changes preserve existing output formats and current SOL ExecBench semantics.
3. Docs continue to warn that AMD-native roofline or performance interpretation must be defined before stronger AMD hardware performance claims.
4. Tests protect existing score output behavior or document intentional internal-only changes.

### Phase 13: Public Contract Compatibility Guardrails

**Goal:** Prove that v1.2 internal improvements did not break supported public
schemas, CLI usage, examples, trace JSONL contracts, or CDNA 3 validation
deferral language.

**Requirements:** COMPAT-01, COMPAT-02, COMPAT-03, CDNADEF-01, CDNADEF-02

**Success Criteria:**
1. Schema tests cover representative supported `definition.json`, `workload.jsonl`, and `solution.json` inputs.
2. CLI and trace-output tests verify existing invocation patterns and trace JSONL contracts remain stable.
3. Example tests or docs checks verify public examples stay valid at documented paths and formats.
4. Docs and planning artifacts continue to state that CDNA 3 real hardware validation is deferred.
5. No updated artifact claims a full `gfx94*` hardware-validation pass without recorded real hardware evidence.

## Requirement Coverage

| Requirement | Phase |
|-------------|-------|
| HARV-01 | Phase 10 |
| HARV-02 | Phase 10 |
| HARV-03 | Phase 10 |
| DIAG-01 | Phase 11 |
| DIAG-02 | Phase 11 |
| DIAG-03 | Phase 11 |
| SCORE-01 | Phase 12 |
| SCORE-02 | Phase 12 |
| SCORE-03 | Phase 12 |
| COMPAT-01 | Phase 13 |
| COMPAT-02 | Phase 13 |
| COMPAT-03 | Phase 13 |
| CDNADEF-01 | Phase 13 |
| CDNADEF-02 | Phase 13 |

**Coverage:** 14/14 v1.2 requirements mapped.

---
*Roadmap created: 2026-05-22*
*Last updated: 2026-05-22 after autonomous execution*
