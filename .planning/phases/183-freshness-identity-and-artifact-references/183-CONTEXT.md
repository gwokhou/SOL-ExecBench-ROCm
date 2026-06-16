# Phase 183 Context: Freshness Identity and Artifact References

## Goal

Attach freshness identity and compact artifact citations to the optional
`sol_execbench.agent_feedback.v1` sidecar so HIP consumers can reject stale or
mismatched feedback without treating the sidecar as benchmark authority.

## Requirements

- IDEN-01: Sidecars include trace path, generated timestamp, SOL contract
  version, optional target/run identity, optional candidate/source hashes, and
  referenced artifact checksums.
- IDEN-02: Artifact citations use compact path/checksum references for
  trace-adjacent diagnostic inputs without exposing raw profiler directories or
  unstable absolute temporary paths.
- IDEN-03: Validator helpers classify stale or identity-mismatched sidecars as
  diagnostic stale states while leaving canonical trace validity unchanged.

## Scope

- Extend the agent-feedback schema with identity, freshness validation, and
  artifact citation models.
- Wire CLI sidecar writing so generated feedback cites the canonical trace and
  any environment/profile/static-evidence sidecars that were written.
- Add CPU-safe tests for identity, checksums, compact paths, and stale/current
  validation.

## Non-Goals

- No canonical Trace JSONL schema changes.
- No HIP adapter/runtime implementation.
- No authority promotion from feedback freshness state.
