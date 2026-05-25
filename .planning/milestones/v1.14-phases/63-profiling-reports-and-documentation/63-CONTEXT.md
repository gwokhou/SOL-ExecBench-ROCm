# Phase 63 Context: Profiling Reports and Documentation

## Discussion

Users need profiling evidence to be operationally visible without mistaking it
for correctness or scoring data. The docs should show the command, output
locations, common ROCm/Docker requirements, and failure semantics.

The evaluator contract can advertise `profiling.evidence.v1` as optional while
leaving `contract_version` at `1.0`.

## Scope

- Document local and Docker profiling usage.
- Explain diagnostic-only status and nonfatal failure behavior.
- Add tests proving optional profiling does not mutate trace JSONL or required
  contract fields.
