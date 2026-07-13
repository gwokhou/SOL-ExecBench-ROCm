# Phase 64 Plan: Claim Boundary and Researcher Positioning

## Objective

Create a central claim-boundary document and tests that make the research
preview's claim language explicit and hard to accidentally overstate.

## Tasks

1. Add `docs/user/CLAIMS.md` with allowed claim levels, unsupported claims, and
   upgrade evidence rules.
2. Add focused tests for the new claims document.
3. Verify the new tests pass.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py -q`

