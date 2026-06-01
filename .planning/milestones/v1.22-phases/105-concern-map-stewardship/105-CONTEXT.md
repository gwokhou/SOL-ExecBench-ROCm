# Phase 105: Concern Map Stewardship - Context

**Gathered:** 2026-06-01
**Status:** Ready for planning

## Phase Boundary

Maintainers can use `CONCERNS.md` as a reliable status map for v1.22 concern
closure and future deferred work.

This phase covers DOCS-01 through DOCS-03:

- Preserve milestone-management context for fixed, narrowed, still actionable,
  accepted, and externally deferred concerns.
- Update each v1.22 in-scope concern with evidence from Phases 100-104.
- Keep out-of-scope items explicit: CDNA3, MI300X, CDNA4 validation,
  paper-scale parity, leaderboard readiness, and complete hard sandboxing.

## Current Concern Map State

`CONCERNS.md` still describes several pre-v1.22 issues as active:

- Dataset runner monolith and text rewriting workaround were narrowed by Phase
  100 via `core/dataset/runner.py` helpers and focused runner tests.
- Reference timing failure and eval-driver framing were fixed/narrowed by Phase
  101 via importable timing helpers, explicit reference timing diagnostics, and
  noisy-output JSONL tests.
- Regex-only reward-hack review was narrowed by Phase 102 via AST-aware Python
  review, broader bypass tests, and structured blocking evidence.
- Scoring fixture and static evidence artifact-manifest gaps were narrowed by
  Phase 103.
- Dependency, closure provenance, and marker guardrail gaps were narrowed by
  Phase 104.

Externally deferred items remain valid and must not be marked fixed:

- Full CDNA3, MI300X, and CDNA4 validation.
- Full 235-problem paper-scale parity and upstream SOLAR equivalence.
- Leaderboard readiness or hosted public service operation.
- Complete hard sandboxing for multi-tenant adversarial submissions.

## Implementation Direction

Update `.planning/codebase/CONCERNS.md` with a short v1.22 status ledger and
per-concern status notes. Avoid overstating fixes: most large modules remain
large or hardware-deferred, but the specific v1.22 guardrails and seams now have
evidence.

## Verification Targets

- `rg -n "v1.22|Status|Deferred|CDNA3|MI300X|CDNA4|hard sandbox|leaderboard|paper-scale" .planning/codebase/CONCERNS.md`
- `git diff --check .planning/codebase/CONCERNS.md`

