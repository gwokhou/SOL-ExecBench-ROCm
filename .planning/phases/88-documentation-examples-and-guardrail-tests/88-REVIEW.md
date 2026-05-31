---
phase: 88-documentation-examples-and-guardrail-tests
reviewed: 2026-05-31T12:22:01Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - docs/v1_19_evidence_guide.md
  - docs/CLAIMS.md
  - docs/TESTING.md
  - docs/RESEARCHER-GUIDE.md
  - docs/examples/v1_19_evidence/README.md
  - docs/examples/v1_19_evidence/execution_closure.demo.json
  - docs/examples/v1_19_evidence/paper_denominator.demo.json
  - docs/examples/v1_19_evidence/paper_denominator.demo.md
  - docs/examples/v1_19_evidence/matrix_schema_export.demo.json
  - docs/examples/v1_19_evidence/matrix_diff.demo.json
  - docs/examples/v1_19_evidence/matrix_diff.demo.md
  - docs/examples/v1_19_evidence/amd_bound_sanity.demo.json
  - docs/examples/v1_19_evidence/amd_bound_sanity.demo.md
  - tests/sol_execbench/test_research_release_docs.py
  - tests/sol_execbench/test_v1_19_evidence_examples.py
  - tests/sol_execbench/test_public_contract_guardrails.py
  - .planning/phases/88-documentation-examples-and-guardrail-tests/88-CONTEXT.md
  - .planning/phases/88-documentation-examples-and-guardrail-tests/88-01-PLAN.md
  - .planning/phases/88-documentation-examples-and-guardrail-tests/88-01-SUMMARY.md
  - .planning/phases/88-documentation-examples-and-guardrail-tests/88-02-PLAN.md
  - .planning/phases/88-documentation-examples-and-guardrail-tests/88-02-SUMMARY.md
findings:
  critical: 0
  warning: 2
  info: 0
  total: 2
status: issues_found
---

# Phase 88: Code Review Report

**Reviewed:** 2026-05-31T12:22:01Z
**Depth:** standard
**Files Reviewed:** 21
**Status:** issues_found

## Summary

Reviewed the Phase 88 public guide, linked docs, demo evidence fixtures, guardrail tests, and phase planning context for claim-boundary wording, demo-vs-validation ambiguity, absolute path/sensitive-log exposure, test gaps, and canonical contract/primary CLI risks.

The implementation keeps canonical schemas, primary CLI, score semantics, evaluator semantics, and hardware-validation scope unchanged. The demo fixtures are clearly marked demo-only/diagnostic-only and avoid real hardware/performance payloads. Two documentation/test-boundary issues remain.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: Conditional wording weakens the "cannot prove" paper-validation boundary

**Classification:** WARNING
**File:** `docs/v1_19_evidence_guide.md:69`
**Issue:** In the "What it cannot prove" sections, the execution-closure bullet says "no full 235-problem paper validation unless all 235 scoped paper problems are actually accounted for with required evidence", and the paper-denominator bullet says "no full 235-problem paper validation when any denominator records or required evidence are absent" at line 105. Both are under a "cannot prove" heading, but the conditional wording creates a claim-boundary loophole: a reader can infer that execution closure or denominator reports alone become full paper validation once the denominator is complete. That contradicts the stronger Phase 88 boundary that sidecars/reports remain diagnostic and do not upgrade paper, score, leaderboard, or hardware authority by themselves.
**Fix:** Make these bullets unconditional and point upgrades back to a separate evidence bundle, without adding any new hardware validation requirement. For example:

```markdown
- no full 235-problem paper validation by this sidecar/report alone; a paper-validation claim requires the separately reviewed complete evidence bundle described in `docs/CLAIMS.md`
```

Apply the same wording style to both execution closure and paper denominator sections so "what it cannot prove" cannot be read as a conditional claim grant.

### WR-02: Absolute-path guardrail is scoped to examples while the central guide publishes `/tmp` command snippets

**Classification:** WARNING
**File:** `docs/v1_19_evidence_guide.md:50`
**Issue:** The Phase 88 fixture tests explicitly reject `/home/`, `/tmp/`, and `/var/` in `docs/examples/v1_19_evidence/*`, but the central researcher-facing guide uses `UV_CACHE_DIR=/tmp/uv-cache` in every command block. This is not a secret leak, but it is still an absolute local path in the primary public guide, and the current tests would not catch it because `test_v1_19_example_wording_repeats_demo_and_negative_boundaries` only scans the examples directory. That leaves the absolute-path/public-doc boundary unenforced exactly where most readers will copy commands.
**Fix:** Either remove `UV_CACHE_DIR` from the public guide snippets or use a relative demo cache path such as `.cache/uv` / `out/v1_19_demo/uv-cache`, then add a focused test that scans `docs/v1_19_evidence_guide.md` plus the example fixtures for disallowed absolute local path prefixes and raw `stdout`/`stderr` bodies. Keep the test CPU-safe and do not broaden it to hardware validation.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_v1_19_evidence_examples.py tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_primary_cli_does_not_expose_v1_19_paper_denominator_options tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_amd_bound_sanity_fields_remain_sidecar_only tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_amd_bound_sanity_does_not_enter_amd_score_contracts tests/sol_execbench/test_public_contract_guardrails.py::test_primary_cli_does_not_expose_v1_19_amd_bound_sanity_options tests/sol_execbench/test_public_contract_guardrails.py::test_phase88_example_docs_keep_v1_19_surfaces_sidecar_only -q` - 24 passed.

---

_Reviewed: 2026-05-31T12:22:01Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
