---
status: complete
quick_id: 260613-merge-no-hip-runtime
slug: merge-no-hip-runtime-rdna4-artifacts
description: Promote the latest kernel-only RDNA4 profiler closure coverage into the canonical merged artifacts
created_at: 2026-06-13T18:30:00+08:00
---

# Quick Task 260613-merge-no-hip-runtime

## Goal

Merge the latest `no-hip-runtime-24` profiler timing closure output into the
canonical RDNA4 re-evaluation merged artifact directory.

## Scope

- Promote the recomputed coverage from
  `out/rdna4-validation-reeval-20260613-latest/coverage-with-no-hip-runtime-24/`.
- Keep canonical merged artifacts internally consistent.
- Do not change benchmark logic or rerun GPU profiling.

