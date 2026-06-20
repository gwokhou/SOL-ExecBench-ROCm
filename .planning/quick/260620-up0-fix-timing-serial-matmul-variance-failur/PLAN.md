---
status: in_progress
created: 2026-06-20
---

# Fix timing serial matmul variance failures

## Goal

Make ROCm device-event timing measure only the queued device work and make the
variance guardrail match its documented trimmed-spread intent.

## Tasks

- Record the end device event immediately after the benchmarked callable.
- Update the matmul variance test to use a trimmed ratio instead of raw
  max/min.
- Give the variance guardrail enough warmup to measure steady state instead of
  ROCm auto-clock ramp-up.
- Re-run the failing timing test and the broader timing serial group.
