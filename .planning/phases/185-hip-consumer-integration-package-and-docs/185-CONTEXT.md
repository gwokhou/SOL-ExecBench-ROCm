# Phase 185 Context: HIP Consumer Integration Package and Docs

## Goal

Provide HIP Playground with stable SOL agent-feedback fixtures, consumer mapping
guidance, and CPU-safe tests for adapter/runtime implementation.

## Requirements

- FIXT-01: Fixtures cover valid, missing/unavailable, malformed, stale, partial,
  and contradictory-authority feedback sidecar cases.
- FIXT-02: Fixture docs explain how HIP should map SOL bottleneck,
  recommendation, limitation, and citation fields into closed prompt-safe
  consumer taxonomies, with unknown values downgraded safely.
- FIXT-03: CPU-safe tests verify generated fixtures and example sidecars remain
  deterministic and contain no raw profiler dump content, full source, raw
  trace rows, or absolute temporary paths.

## Scope

- Add fixture JSON under `tests/sol_execbench/fixtures/agent_feedback/`.
- Add HIP-facing docs in `docs/agent_feedback_sidecar.md`.
- Add tests that validate fixture parsing, negative fixture handling, freshness
  downgrade, prompt-safety, and documentation coverage.
