# Phase 71 Plan: Dynamic Probe and Routing Policy

## Scope

1. Implement bounded `probe_toolchain_tool`.
2. Implement `build_toolchain_routing_report`.
3. Add `sol-execbench toolchain --json` and `--list-registry`.
4. Keep trace JSONL unchanged.
5. Add tests for available, migrated, unsupported-artifact, and planned static
   routes.

## Acceptance Criteria

- Routing selects an available profiling tool when present.
- Migrated tools produce fallback metadata.
- Static routes are represented as planned/candidate/unavailable, not executed.
- CLI emits JSON without requiring a benchmark problem.
