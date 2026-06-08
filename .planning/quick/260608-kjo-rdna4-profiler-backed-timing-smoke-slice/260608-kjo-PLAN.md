---
quick_id: 260608-kjo
slug: rdna4-profiler-backed-timing-smoke-slice
status: in_progress
created_at: 2026-06-08
---

# Quick Task 260608-kjo: RDNA4 Profiler-Backed Timing Smoke Slice

## Goal

Add a small, runnable RDNA4 timing smoke entry point that targets a Triton
example and records whether `rocprofv3` produced kernel-activity timing
sidecars. This should improve the path from "all PyTorch fallback timing" to a
bounded profiler-backed timing slice without upgrading full RDNA4 validation
claims.

## Tasks

1. Add a smoke script.
   - Files: `scripts/run_rdna4_profiler_timing_smoke.py`
   - Action: run a bounded Triton RMSNorm workload through
     `collect_source_timing_evidence()`, write timing and summary artifacts, and
     fail unless profiler-backed timing is collected or fallback is explicitly
     allowed.
   - Verify: CPU-safe unit tests with fake `rocprofv3` runner.

2. Add tests.
   - Files: `tests/sol_execbench/test_rdna4_profiler_timing_smoke.py`
   - Action: verify default Triton command selection, limited workload staging,
     `profiler_collected=true` summary, and fallback failure behavior.
   - Verify: focused pytest.

3. Record GSD quick summary and state.
   - Files: `.planning/quick/260608-kjo-rdna4-profiler-backed-timing-smoke-slice/260608-kjo-SUMMARY.md`,
     `.planning/STATE.md`
   - Action: summarize implementation and verification.
