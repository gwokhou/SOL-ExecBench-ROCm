# Research: Features for v1.36 SOL Agent Feedback Sidecar Producer

**Date:** 2026-06-15

## Table Stakes

- Optional sidecar file named `trace.jsonl.agent-feedback.json` when an output
  trace path is available.
- Schema includes diagnostic-only authority flags and rejects contradictory
  authority claims.
- Contract advertises optional capabilities for agent-feedback/profile-summary
  so HIP can detect support without breaking older SOL versions.
- Payload exposes bounded bottleneck, recommendation, limitation, freshness,
  and artifact citation fields.
- Missing, skipped, unavailable, partial, malformed-source, and stale-source
  states are explicit and do not alter canonical evaluation status.
- Fixtures cover success and invalid states for HIP adapter tests.

## Differentiators

- Recommendations are source-cited and bounded to ROCm/HIP-safe next-experiment
  guidance rather than free-form prompt text.
- Freshness identity ties the sidecar to trace path, run target, source/candidate
  hash when available, SOL contract version, and artifact checksums.
- Existing evaluation-stability and profiling reason codes can be summarized
  without making profiler output mandatory.

## Deferred

- Multi-source confidence scoring for bottleneck classification.
- Parsing full profiler counter dumps into a stable hardware-counter taxonomy.
- Using feedback for candidate ranking or score authority.
- Cross-run feedback accumulation.
