---
phase: 144
title: RDNA4 derived sidecar exclusion retry
---

# Phase 144 Context

Phase 140 completed RDNA4 AMD-derived evidence with temporary long-tail
sidecar exclusions so memory-heavy derived generation would not take down Codex
or the calling shell. The retry input is:

- `out/rdna4-derived-reports/rdna4-derived-long-tail-exclusions.json`

That file contains 10 temporary exclusion records across 8 problems. Those
records account for the remaining temporary derived sidecar gaps reported by
the Phase 140 evidence bundle.

Phase 144 should retry those excluded problems without passing the exclusion
config, but every retry must remain isolated and memory-capped. The user
explicitly observed repeated memory/swap exhaustion; use `systemd-run --user`
or equivalent memory limits, persist logs/status files, and poll rather than
owning heavy child processes from Codex.

The retry outputs should be written under a new v1.31 evidence root so Phase
140 artifacts remain preserved:

- `out/rdna4-derived-retry-v131/`
