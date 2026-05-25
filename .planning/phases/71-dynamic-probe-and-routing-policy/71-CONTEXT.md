# Phase 71 Context: Dynamic Probe and Routing Policy

## Objective

Combine static registry facts with bounded dynamic probes so routing can select
available tools or explain unavailable, unsupported, migrated, planned, or
failed states.

## Decisions

- Dynamic probes are timeout-bounded and injectable for tests.
- Selected/fallback/reason fields are emitted in sidecar-like metadata only.
- Static Kernel Evidence remains deferred to v1.17.
