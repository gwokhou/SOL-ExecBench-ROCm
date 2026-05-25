# Phase 69 Plan: Toolchain Inventory and Lifecycle Model

## Scope

1. Add tool lifecycle enum and registry entry model.
2. Seed built-in registry with current, historical, migrated, planned, and
   candidate tools.
3. Preserve source references and replacement tool IDs.
4. Add tests for lifecycle and future static tool representation.

## Acceptance Criteria

- Registry includes `rocprofv3`, `rocprofv3-avail`, ROCm Systems,
  `rocprofiler-systems`, `rocminfo`, `rocm_agent_enumerator`, RGA,
  `llvm-objdump`, `roc-objdump`, and `readelf`.
- Deprecated/migrated tools are explicit.
- Static tools are present but not implemented as static evidence in v1.16.
