---
phase: 88-documentation-examples-and-guardrail-tests
reviewed: 2026-05-31T13:00:33Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - docs/v1_19_evidence_guide.md
  - tests/sol_execbench/test_research_release_docs.py
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Phase 88: Targeted Re-Review Report

**Reviewed:** 2026-05-31T13:00:33Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** clean

## Summary

Targeted re-review of latest commit `d26bd65` (`#88 - Fix remaining v1.19 guide command gaps`) focused only on `docs/v1_19_evidence_guide.md`, `tests/sol_execbench/test_research_release_docs.py`, and the two remaining integration-check blockers. No source or hardware validation scope was expanded. `.omc/` was left untouched.

Verdict: PASS. Both remaining guide command gaps are closed, and no new Critical/Blocker or Warning findings were found in the requested scope.

## Narrative Findings (AI reviewer)

All reviewed files meet the targeted quality bar. No findings.

## Audit Blocker Closure

### BL-01: `run_dataset.py` guide example used non-existent `--output-dir`

**Status:** closed / PASS
**Files:** `docs/v1_19_evidence_guide.md:50`, `docs/v1_19_evidence_guide.md:53`, `tests/sol_execbench/test_research_release_docs.py:294`, `tests/sol_execbench/test_research_release_docs.py:295`
**Result:** The execution closure example now invokes `scripts/run_dataset.py` with the real `--output out/v1_19_demo/run-dataset` option. `scripts/run_dataset.py` defines `-o` / `--output` for the output directory and does not define `--output-dir`. The docs guardrail now asserts the correct example and rejects the stale `--output-dir out/v1_19_demo/run-dataset` spelling.

### BL-02: `export_matrix_schema.py` guide example used `--output-dir` without `--model all`

**Status:** closed / PASS
**Files:** `docs/v1_19_evidence_guide.md:122`, `docs/v1_19_evidence_guide.md:123`, `docs/v1_19_evidence_guide.md:124`, `tests/sol_execbench/test_research_release_docs.py:296`, `tests/sol_execbench/test_research_release_docs.py:297`
**Result:** The Matrix schema export example now includes `--model all` before `--output-dir out/v1_19_demo/matrix-schema`. `scripts/export_matrix_schema.py` requires `--model all` for directory export and rejects `--output-dir` for single-model exports. The docs guardrail now asserts both the required model selector and the schema output directory.

## Verification

- `uv run pytest tests/sol_execbench/test_research_release_docs.py -q` - 15 passed.
- `uv run ruff check tests/sol_execbench/test_research_release_docs.py` - passed.

GPU, Docker, hardware-marker tests, dependency relocking, and `.omc/` inspection were intentionally not run for this targeted re-review.

---

_Reviewed: 2026-05-31T13:00:33Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
