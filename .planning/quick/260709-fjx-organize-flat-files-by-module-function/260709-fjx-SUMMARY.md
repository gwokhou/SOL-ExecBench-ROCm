---
status: complete
---

# Summary

- Grouped flat `core/bench` feature clusters into focused subpackages.
- Grouped flat `core/scoring` feature clusters into focused subpackages.
- Preserved legacy module imports through compatibility facades and package entry points.
- Validation not run by request policy; run targeted module-boundary tests if desired.

- Kept high-inbound scoring model modules at legacy paths to satisfy architecture guardrails.
