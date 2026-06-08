---
phase: 163
status: complete
completed_at: "2026-06-08"
---

# Phase 163 Summary: RDNA4 Denominator Policy Hardening

## Result

Added an explicit RDNA4 `gfx1200` 16GB denominator policy. The policy locks the
Phase 162 boundary: `reference_oom_blocked` problems remain accounted in the
235-problem denominator but do not count as profiler-backed timing or full
validation pass evidence.

## Implementation

- Added `docs/internal/RDNA4-DENOMINATOR-POLICY.md`.
- Linked the policy from `docs/CLAIMS.md`.
- Added CPU-safe regression coverage proving reference OOM accounting does not
  satisfy full profiler-backed timing coverage.

## Verification

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_profiler_timing_coverage.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check docs/internal/RDNA4-DENOMINATOR-POLICY.md docs/CLAIMS.md tests/sol_execbench/test_profiler_timing_coverage.py`
