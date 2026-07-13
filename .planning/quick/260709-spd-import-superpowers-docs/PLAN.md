---
quick_id: 260709-spd
slug: import-superpowers-docs
status: in_progress
created_at: "2026-07-09T00:00:00Z"
---

# Quick Task: Import Superpowers Docs Into GSD Planning

## Goal

Convert the legacy `docs/internal/superpowers/` planning and design documents into GSD
planning artifacts and merge them under `.planning/` without changing active
v1.38 roadmap phase scope.

## Plan

1. Inventory `docs/internal/superpowers/plans/` and `docs/internal/superpowers/specs/`.
2. Convert implementation plans into GSD quick task directories with source
   frontmatter and preserved original content.
3. Convert design specs into GSD research notes under `.planning/research/`.
4. Add an index that maps every source document to its new GSD location.
5. Update `STATE.md` quick task tracking and verify all source files are
   represented in `.planning`.
