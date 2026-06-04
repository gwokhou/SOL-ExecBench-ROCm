---
quick_id: 260603-ffd
slug: gsd-health
status: complete
created: 2026-06-03
---

# Quick Task: Auto-fix GSD health leftovers

## Goal

Resolve the remaining non-repairable `$gsd-health` warnings without deleting
planning history.

## Plan

1. Add a phase archive index for completed v1.27 phases 123-126 to
   `.planning/ROADMAP.md` so the health validator can match archived phase
   directories to roadmap entries.
2. Move root-level validation handoff documents into `.planning/milestones/`
   because they are milestone/archive material, not canonical `.planning/`
   root artifacts.
3. Re-run `$gsd-health` and record the final result.
