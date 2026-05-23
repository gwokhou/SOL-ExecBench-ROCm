---
phase: 56-parity-gap-reporting-and-evidence-review
plan: 02
status: complete
completed: 2026-05-23
---

# Plan 56-02 Summary - CLI And Evidence Completeness

## Delivered

- Added `scripts/report_parity_gaps.py` as a thin CLI for JSON and Markdown
  report generation.
- Added deterministic Markdown rendering with suite, category, blocker,
  evidence, source, and claim-boundary sections.
- Evidence completeness now distinguishes trace, timing, AMD-native score, AMD
  SOL, and SOLAR derivation evidence.

## Verification

- `uv run pytest tests/sol_execbench/test_parity_gap_report.py -n 0 -x`
- `uv run --with ruff ruff check src/sol_execbench/core/dataset/parity_gap.py src/sol_execbench/core/dataset/__init__.py scripts/report_parity_gaps.py tests/sol_execbench/test_parity_gap_report.py`
