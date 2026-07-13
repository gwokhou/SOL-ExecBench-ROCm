---
status: complete
completed: 2026-05-31
---

# Fix v1.20 Audit Doc Wiring Tech Debt Summary

Closed DOC-WIRE-01 from `.planning/v1.20-MILESTONE-AUDIT.md`:

- Added `--amd-sol-report` and `--solar-derivation` to the public
  `scripts/report_consistency.py` example in `docs/internal/v1_20_evidence_quality_guide.md`.
- Added a docs regression requiring the consistency example to include both
  AMD SOL and SOLAR refs.
- Updated the v1.20 milestone audit status from `tech_debt` to `passed`.

Verification:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_v1_20_evidence_quality_docs.py -q`
