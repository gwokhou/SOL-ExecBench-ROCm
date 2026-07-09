---
status: complete
date: 2026-07-09
---

# Summary

Removed the remaining backward-compatibility cleanup targets:

- Deleted flat bench/scoring compatibility facade modules after rewriting imports to canonical package modules.
- Moved remaining model modules into their owning packages and updated public package exports.
- Removed legacy AMD SOL fallback work estimation.
- Removed legacy SOLAR derivation sidecar parsing that accepted payloads without `coverage_summary` and `aggregate_status`.
- Removed the unused legacy clock preset API and old ignored parameters on clock locking helpers.
- Fixed stale console/package CLI exports and updated provenance/residue guardrails for the new layout.

Verification:

- `uv run pytest tests/` -> 1885 passed, 41 skipped.
- `uv run --with ruff ruff check .` -> passed.
- Final targeted scan found no explicit old compatibility facade/fallback symbols among source, tests, scripts, pyproject, or provenance manifest.
