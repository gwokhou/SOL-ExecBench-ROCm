# Phase 62 Context: rocprofv3 Artifact Lifecycle

## Discussion

The v1.14 artifact path should make profiler output durable when users provide
`--output`, while keeping no-output runs consistent with existing staging
behavior. The preferred profiler output is `rocpd` because it preserves richer
inspection data; CSV outputs should still be registered when present.

Failure metadata must include stdout/stderr tails, exit status, skipped reason,
failed reason, and any partial artifacts. These records are diagnostic-only and
must not become score authority.

## Scope

- Register `rocpd`, CSV trace, counter, agent-info, JSON, and unknown artifacts.
- Write profile sidecar metadata next to trace JSONL.
- Preserve partial artifacts on profiler failure.
