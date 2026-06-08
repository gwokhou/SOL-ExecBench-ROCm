---
phase: 146
title: RDNA4 failed workload triage and claim reassessment
---

# Phase 146 Context

Phase 138 recorded 146 failed RDNA4 workload records in the bounded ready
subset. Phase 145 removed the ambiguity around the 12 `missing_trace` rows by
classifying them as `gpu_oom_no_trace`.

Phase 146 closes the remaining failure-triage and claim-boundary work:

- Group all 146 failed workloads by failure class.
- Preserve representative examples, likely root cause, and next action.
- Reassess public and planning claims without upgrading beyond bounded
  `gfx1200` ready-subset evidence.

Primary inputs:

- `out/rdna4-full-dataset/run/execution_closure.json`
- `out/rdna4-full-dataset/run/**/traces.json`
- `out/rdna4-missing-trace-triage-v131/phase145-missing-trace-triage.json`

