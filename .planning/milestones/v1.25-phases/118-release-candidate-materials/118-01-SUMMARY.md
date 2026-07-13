---
phase: 118
phase_name: "Release Candidate Materials"
plan: 1
status: complete
completed_at: "2026-06-01"
requirements: [REL-01, REL-02, REL-03]
---

# Phase 118 Summary

## Completed Work

- Added `docs/internal/v1_25_prerelease_checklist.md` with a maintainer flow from clean
  tree to annotated release-candidate tag and push.
- Updated `docs/internal/v1_25_release_notes.md` with deferred claims and public entry
  points.
- Updated `README.md` with v1.25 prerelease materials, claims, and first-run
  troubleshooting navigation.
- Added CPU-safe release-material guardrails in
  `tests/sol_execbench/test_research_release_docs.py`.

## Evidence

- Plan/context commit: `0abb751 docs(118): plan release candidate materials`
- Implementation commit: `970af21 docs: add v1.25 release candidate materials`
- Review: `.planning/phases/118-release-candidate-materials/118-REVIEW.md`

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_release_candidate_validation.py -q
```

Result: `78 passed in 11.13s`.

## Requirement Closure

- REL-01: Complete. Maintainers have a prerelease checklist from clean tree to
  tagged release candidate.
- REL-02: Complete. Release notes summarize shipped capability, validation
  evidence, known limitations, and deferred claims.
- REL-03: Complete. Public docs point to support matrix, claims, researcher
  guide, timing semantics, release validation, and first-run troubleshooting.
