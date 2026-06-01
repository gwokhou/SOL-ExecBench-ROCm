# Phase 113 Context: Deterministic Dataset Sharding Path

## Objective

Define the first deterministic dataset sharding semantics without changing the
default dataset CLI behavior.

## Scope

- Add core helpers that can plan workload shards with stable shard identifiers
  and one trace path per shard.
- Add merge helpers that preserve workload ordering, detect duplicate trace
  ownership, and report incomplete shards.
- Cover the design with CPU-safe tests.

## Out Of Scope

- Adding `scripts/run_dataset.py --shard-*` flags.
- Running multiple processes.
- Live ROCm validation.
