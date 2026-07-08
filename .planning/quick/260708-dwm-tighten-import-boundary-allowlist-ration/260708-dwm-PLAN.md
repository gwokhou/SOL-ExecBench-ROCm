---
status: in-progress
date: 2026-07-08
---

# Quick Task 260708-dwm: Tighten Coupling Boundaries

## Scope

Make import-boundary allowlist entries self-documenting and move persisted AMD
score sidecar parsing out of the AMD score report orchestration module.

## Tasks

1. Add failing tests for allowlist rationales and AMD score report parser
   helper placement.
2. Move minimal persisted sidecar parsing into a dedicated scoring helper
   module.
3. Run focused AMD score report and boundary tests plus Ruff.
