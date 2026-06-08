---
phase: 145
title: RDNA4 missing trace root-cause triage
---

# Phase 145 Context

Phase 138 completed the RDNA4 ready-subset run with 12 explicit
`missing_trace` workload closure records. Those records were intentionally kept
visible so failed workloads would not be silently dropped from the denominator.

Phase 145 resolves the remaining ambiguity: determine whether the 12 rows are
missing artifacts, stale closure records, or concrete execution failures that
ended before a trace could be emitted.

Primary inputs:

- `out/rdna4-full-dataset/run/execution_closure.json`
- `out/rdna4-full-dataset/run/**/traces.json`
- `out/rdna4-full-dataset/run/**/*_cli.log`
- `.planning/milestones/v1.30-phases/138-rdna4-full-dataset-execution-and-denominator-closure/138-SUMMARY.md`

The validation host remains the recorded RDNA4 `gfx1200` machine with a 16 GiB
GPU memory pool.

