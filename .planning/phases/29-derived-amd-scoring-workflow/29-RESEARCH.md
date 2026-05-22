# Phase 29 Research: Derived AMD Scoring Workflow

**Phase:** 29 - Derived AMD Scoring Workflow
**Researched:** 2026-05-22
**Status:** Ready for planning

## Current Implementation

`amd_score.py` has guarded workload and suite report models but only exposes
manual builders. `scripts/run_dataset.py` writes canonical traces and a dataset
summary but does not yet write AMD-native score reports.

## Recommended Approach

Add workflow helpers that bridge existing trace/evidence inputs to the existing
score dataclasses. Then add an optional dataset runner report path that builds
derived AMD-native scores from problem definitions, workloads, traces, baseline
latency fields, and generated SOL bounds.

## Guardrails

- Keep score reports derived and separate from trace JSONL.
- Preserve `sol_score()`.
- Treat missing evidence as guarded/unscored.
- Preserve claim guardrails for unsupported operations, unvalidated hardware,
  and CDNA3 no-claim status.
