---
quick_id: 260613-tc4
slug: rdna4-profiler-reference-patch-override-
status: in_progress
created_at: 2026-06-13T13:07:21.467Z
---

# Quick Task: Mark RDNA4 Reference Patch Override Evidence

## Goal

Make profiler timing sidecars and coverage ledgers explicitly mark timing evidence
collected with an RDNA4 reference implementation override.

## Scope

- Keep the existing scoped override for `L2/035_convnextv2_block_with_grn`.
- Record override metadata when the replacement timing batch applies it.
- Surface the metadata in coverage evidence summaries and blocker ledgers.
- Add regression coverage.

## Non-Goals

- Do not add new reference overrides.
- Do not change dataset benchmark files.
- Do not claim the override is original-reference dispatch timing.
