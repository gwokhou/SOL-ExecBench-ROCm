# Phase 70 Plan: Capability Registry Schema

## Scope

1. Add enums for evidence level, artifact type, lifecycle, and status.
2. Add capability, request, decision, probe, and report models.
3. Add schema version `sol_execbench.toolchain_routing.v1`.
4. Add tests for serialization and authority boundaries.

## Acceptance Criteria

- Registry and routing payloads are JSON serializable.
- Status vocabulary includes all required v1.16 states.
- Reports carry diagnostic-only authority flags.
