# Phase 30 Research: Compatibility and Claim Guardrails

**Phase:** 30 - Compatibility and Claim Guardrails
**Researched:** 2026-05-22
**Status:** Ready for planning

## Current State

v1.6 added derived artifacts and additive workflow surfaces:

- AMD SOL coverage summaries in bound artifacts.
- Live `rocprofv3` collection adapter with fallback metadata.
- Dataset runner opt-in `--amd-score-report` derived score report.

The primary benchmark contract remains canonical trace JSONL emitted by
`sol-execbench`.

## Guardrail Targets

- Existing trace models and workload-only trace parsing.
- Existing primary CLI help/defaults.
- Derived artifact markers and evidence references.
- CDNA3 validation deferral wording.
- NVIDIA B200/upstream SOLAR/leaderboard no-claim wording.
- Hardware model source/confidence/validation evidence.

## Recommended Work

Add v1.6-focused public contract tests and update stale v1.5 claim wording in
CDNA3 score warnings. Keep changes narrowly scoped.
