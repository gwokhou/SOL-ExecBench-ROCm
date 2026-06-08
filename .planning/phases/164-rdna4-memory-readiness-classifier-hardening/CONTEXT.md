# Phase 164 Context: RDNA4 Memory Readiness Classifier Hardening

## Goal

Generalize memory/readiness classification so RDNA4 reports distinguish
reference OOM, input-generation OOM, user-solution OOM, timeout, profiler
blocked, and correctness/runtime failures deterministically.

## Depends On

- Phase 163: RDNA4 denominator policy hardening

## Scope

- Harden classifier code and report fields around memory footprint blockers.
- Preserve scheduler/resume visibility so unsafe full-problem profiler retries
  are not selected by default.
- Add regression fixtures for the 10 diagnosed partial profiler targets.

## Primary Deliverable

- Classifier implementation and CPU-safe regression tests.
