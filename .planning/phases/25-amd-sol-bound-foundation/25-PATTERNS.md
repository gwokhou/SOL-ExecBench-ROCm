# Phase 25: AMD SOL Bound Foundation - Pattern Map

**Mapped:** 2026-05-22

## File Mapping

| Target File | Role | Closest Analog | Pattern |
|-------------|------|----------------|---------|
| `src/sol_execbench/core/scoring/amd_sol.py` | New graph/work/bound module | `core/reporting.py`, `core/bench/timing_policy.py` | Frozen dataclasses and `to_dict()` evidence objects. |
| `tests/sol_execbench/test_amd_sol_bounds.py` | Unit tests | `test_timing_policy.py` | Pure assertions over small synthetic objects. |
| `docs/analysis.md` | User docs | Existing AMD-native section | Add bound artifact prerequisite language. |

## Guardrails

- No trace schema changes.
- No final score aggregation in Phase 25.
- CDNA3 hardware model entries must be unvalidated.

---
*Pattern map completed: 2026-05-22*
