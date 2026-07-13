---
status: resolved
trigger: "Fix remaining failures from `uv run pytest tests/` after docs update."
created: 2026-06-14
updated: 2026-06-14
---

# Debug Session: fix-remaining-test-failures

## Symptoms

- Expected behavior: `uv run pytest tests/` should pass, apart from expected skips.
- Actual behavior: full suite reported 13 failures and 32 collection errors.
- Error messages: missing root-level script files after scripts moved under `scripts/internal/`, one PyTorch dtype assertion failure, and public docs/planning wording guardrail failures.
- Timeline: observed after running full suite on 2026-06-14.
- Reproduction: run `uv run pytest tests/`.

## Current Focus

- hypothesis: tests still reference legacy root-level report script paths and a few docs assertions need synchronized wording.
- test: run focused failing tests after each fix cluster.
- expecting: missing-script import failures disappear; docs guardrail assertions pass; dtype assertion is either fixed or isolated.
- next_action: inspect failing tests and current script layout.

## Evidence

## Eliminated

## Resolution

- root_cause: tests and compatibility expectations still referenced legacy root-level report script paths after scripts moved under `scripts/internal/`; public guardrail docs/planning missed exact assertion phrases; one README edit removed a required "troubleshooting" navigation keyword.
- fix: added root-level compatibility wrapper scripts that delegate to internal scripts, restored the README keyword while avoiding a dead link, corrected generated docs, and added missing claim-boundary wording.
- verification: `uv run pytest tests/` passed with 1664 passed and 63 skipped.
- files_changed: README.md, docs/user/GETTING-STARTED.md, docs/user/DEVELOPMENT.md, docs/user/CLAIMS.md, .planning/milestones/v1.28-REQUIREMENTS.md, scripts/*.py compatibility wrappers.
