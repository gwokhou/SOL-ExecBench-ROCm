# Phase 60 Research: Diagnostics CLI and Preflight Checks

**Researched:** 2026-05-25
**Status:** Complete

## Existing Hooks

- The root Click command already special-cases `contract`.
- `tests/conftest.py` already has RDNA4/CDNA3 marker support, so Phase 60 does
  not need to invent marker behavior.
- Phase 58 provides snapshot evidence and status vocabulary.

## Recommended Implementation

- Extend the root command dispatcher with `doctor`.
- Add diagnostics models in `core.environment` so the command can emit stable
  JSON.
- Implement PyTorch/HIP smoke checks only when doctor is called.
- Add tests with injected fake diagnostics builders.

