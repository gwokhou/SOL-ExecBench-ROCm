---
phase: 166
nyquist_compliant: true
wave_0_complete: true
validated_at: "2026-06-09T00:11:40+08:00"
---

# Phase 166 Validation

## Result

Nyquist compliant for the scoped profiler-closure deliverable with accepted
current-device blockers.

## Evidence Checked

- `out/rdna4-rocprof-timing-closure-20260608-smoke/`
- `out/rdna4-rocprof-timing-closure-20260608-l1028-sharded/`
- `out/rdna4-rocprof-timing-closure-20260608-l1053-sharded/`
- `out/rdna4-rocprof-timing-closure-20260608-l1053-offset9-retry/`
- `out/rdna4-coverage-recompute-accepted-20260609/`

## Validation

- Profiler closure attempts are recorded.
- Current-device profiler-closure OOM is explicitly classified as
  `profiler_closure_oom_blocked`.
- The accepted blocker rows remain outside the `profiler_backed` numerator.
