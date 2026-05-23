# Phase 52 Plan Check

**Phase:** 52 Dataset Runner And Public Contract Closure  
**Checked:** 2026-05-23  
**Result:** BLOCK

## Executive Result

Phase 52 plans are goal-aligned and mostly complete, but execution should not proceed until the research open questions are marked resolved and the chosen answers are made explicit. The plans otherwise cover the requested runner/report integration, skipped-trace risk, documentation, public contract guardrails, and claim guardrails.

## Goal-Backward Coverage

Goal: Users can run v1.10 derivation through intended reporting surfaces with documentation and guardrails that preserve public contracts and claim boundaries.

| Required outcome | Evidence in plans | Status |
| --- | --- | --- |
| `REPORT-04`: derived reports preserve `amd-native-derived` and include formula, hardware model, coverage, and score eligibility evidence refs | 52-01 lines 19-22, 120-130; 52-02 lines 72-79; 52-03 lines 89-99 | Covered |
| `TEST-04`: public contracts prove canonical schemas, trace JSONL, primary CLI behavior, and public semantics unchanged | 52-01 lines 120-125; 52-03 lines 21-24, 87-99, 183-186 | Covered |
| `TEST-05`: guard against paper parity, B200/Blackwell, hosted leaderboard, and new hardware validation overclaims | 52-02 lines 82-100; 52-03 lines 102-115 | Covered |
| No primary `sol-execbench` CLI changes | 52-01 lines 88-90; 52-03 lines 90-95 | Covered |
| No canonical trace JSONL changes | 52-01 lines 120-125; 52-03 lines 90-95 | Covered |
| No new dependencies | 52-01 lines 159-167; 52-03 lines 140-148 | Covered |
| No hardware/candidate execution | 52-01 lines 103-109, 119-125; 52-02 lines 92-100; 52-03 lines 118-126 | Covered |
| Runner derived artifacts are opt-in | 52-01 lines 19-26, 119-125; 52-02 lines 72-79 | Covered |
| Known risk: skipped passed traces still contribute to requested reports | 52-01 lines 21, 43-46, 136-145 | Covered |

## Plan Structure

| Plan | Wave | Tasks | Files | Dependency status | Structural status |
| --- | ---: | ---: | ---: | --- | --- |
| 52-01 | 1 | 3 | 4 | No deps | Valid |
| 52-02 | 2 | 3 | 2 | Depends on 52-01 | Valid |
| 52-03 | 3 | 3 | 4 | Depends on 52-01, 52-02 | Valid |

`gsd-sdk query verify.plan-structure` reports all three plans valid with required `files`, `action`, `verify`, and `done` fields on every task.

## Findings

### Blockers

1. **[research_resolution] `52-RESEARCH.md` has unresolved Open Questions**
   - Severity: BLOCKER
   - Evidence: `52-RESEARCH.md` lines 388-397 has `## Open Questions` without `(RESOLVED)`, and neither question has an inline `RESOLVED` marker.
   - Why it blocks: the plan-check gate requires all research questions to be resolved before execution. The two questions directly affect implementation choices: whether Phase 52 reads existing SOLAR sidecars or only writes/pass-through generated sidecars, and where report-only internal SOLAR refs live.
   - Required fix: update research to `## Open Questions (RESOLVED)` and mark both questions resolved. Also make the selected answers explicit in 52-01, for example: generated/write-through sidecars are required; read-existing sidecars are included or out of scope; derived refs live in the chosen derived-report-only field.

### Warnings

1. **[verification_derivation] Plan 52-02 includes weak/non-discriminating plan-level verification commands**
   - Severity: WARNING
   - Evidence: `52-02-PLAN.md` lines 141-145 includes `uv run pytest tests/sol_execbench/test_amd_native_score.py -k "docs or equivalence or claim"` and Ruff over markdown files.
   - Risk: the pytest command is likely to select no relevant doc/claim tests, and Ruff does not provide meaningful markdown/documentation validation. The task-level `rg` checks help, and Plan 52-03 adds stronger claim tests, so this is not a phase blocker.
   - Recommended fix: replace the plan-level pytest command with the actual claim/doc guardrail files used in 52-03, or rely on 52-03 for automated docs/claim closure. Keep `rg` checks for exact wording.

## Dimension Results

| Dimension | Result | Notes |
| --- | --- | --- |
| Requirement coverage | PASS | `REPORT-04`, `TEST-04`, and `TEST-05` appear in plan frontmatter and have concrete tasks. |
| Task completeness | PASS | All tasks have files/action/verify/done. |
| Dependency correctness | PASS | 52-01 -> 52-02 -> 52-03 is acyclic and wave-correct. |
| Key links planned | PASS | Runner-to-derivation, runner-to-score, docs-to-runner, and guardrail-to-public-contract links are explicit. |
| Scope sanity | PASS | Each plan has 3 tasks and fewer than 10 files. |
| Verification derivation | WARN | 52-02 plan-level verification needs tightening. |
| Context compliance | PASS | Locked decisions are implemented; deferred ideas are excluded or framed as non-claims. |
| Scope reduction | PASS | No `v1`, `static for now`, stub, or future-deferral reduction of locked decisions found. |
| Architectural tier compliance | PASS | Runner work stays in `scripts/run_dataset.py`; scoring in core scoring; public/claim protection in tests/docs. |
| Nyquist compliance | PASS | `52-VALIDATION.md` exists; every task has automated verification; no watch commands or missing test placeholders. |
| Cross-plan data contracts | PASS | 52-02 depends on 52-01 for option names; 52-03 depends on 52-01/52-02 for final guardrails. |
| AGENTS.md compliance | PASS | Uses pytest/Ruff, no new deps, no hardware requirement, no source edits outside planned GSD execution. |
| Research resolution | BLOCK | Open Questions are not marked resolved. |
| Pattern compliance | PASS | Plans reference existing runner, scoring, sidecar, exact-key, and claim guardrail patterns from `52-PATTERNS.md`. |

## Structured Issues

```yaml
issues:
  - plan: null
    dimension: research_resolution
    severity: blocker
    description: "52-RESEARCH.md has unresolved Open Questions without RESOLVED markers."
    evidence:
      - ".planning/phases/52-dataset-runner-and-public-contract-closure/52-RESEARCH.md:388"
      - ".planning/phases/52-dataset-runner-and-public-contract-closure/52-RESEARCH.md:390"
      - ".planning/phases/52-dataset-runner-and-public-contract-closure/52-RESEARCH.md:394"
    fix_hint: "Mark the Open Questions section RESOLVED and record explicit answers for sidecar read/write scope and derived-report-only ref placement."
  - plan: "52-02"
    dimension: verification_derivation
    severity: warning
    description: "Plan-level docs verification uses likely non-discriminating commands."
    task: null
    evidence:
      - ".planning/phases/52-dataset-runner-and-public-contract-closure/52-02-PLAN.md:144"
      - ".planning/phases/52-dataset-runner-and-public-contract-closure/52-02-PLAN.md:145"
    fix_hint: "Use the actual claim/doc guardrail tests or leave docs verification to Plan 52-03 while keeping exact rg assertions."
```

## Recommendation

BLOCK until the research questions are explicitly resolved. After that revision, the plans are likely executable with only the 52-02 verification warning to clean up.
