---
quick_id: 260709-my4
slug: v1-43-tag-release-release-tag
status: complete
created_at: "2026-07-09T08:31:20.771Z"
---

# Quick Task: Fix v1.43 Release Tag Constant

## Goal

Make the runtime SOL release identity match the actual `v1.43` tag.

## Plan

1. Update `SOL_EXECBENCH_RELEASE` from `v1.42` to `v1.43`.
2. Refresh active tests and HIP-facing current fixtures that assert or encode the
   default release identity.
3. Keep stale fixtures on `v1.42` so freshness checks still cover old-release
   diagnostics.
4. Run focused tests for contract, agent feedback, profile summary, and sidecar
   release identity paths.
