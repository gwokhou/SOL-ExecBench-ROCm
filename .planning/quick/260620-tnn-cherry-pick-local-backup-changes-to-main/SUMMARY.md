---
status: complete
completed: 2026-06-20
---

# Cherry Pick Local Backup Changes To Main Summary

Applied backup commit `a33600b` to the current `main` branch with manual conflict
resolution.

Kept current `main` versions of `.planning/STATE.md` and
`.planning/codebase/*` where the backup contained stale pre-sync mapping
content, then recorded this quick task in `STATE.md`.

Restored the backup's compatibility wrapper scripts under `scripts/`, preserved
the Cookbook README link, kept the no-trace sidecar wording update, and retained
the RDNA4/validation documentation notes that still apply after the remote sync.
