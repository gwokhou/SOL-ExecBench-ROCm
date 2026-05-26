# Phase 76: CLI Sidecar Integration And Reports - Research

**Researched:** 2026-05-26  
**Domain:** Click CLI opt-in, diagnostic sidecar writing, and benchmark-semantics guardrails  
**Confidence:** HIGH

## Summary

Phase 76 can follow existing profiler sidecar patterns in `cli/main.py`: derive
an output directory from the trace path when available, write JSON metadata
beside the trace output, and keep diagnostic collection failures nonfatal. The
static evidence trigger point belongs immediately after HIP/C++ compile success,
because Phase 74 discovery needs the current staging/build tree before cleanup
and Phase 75 extraction uses persisted artifacts from the evidence directory.

## Requirement Mapping

| Requirement | Implementation Direction |
|-------------|--------------------------|
| SKE-CLI-01 | Add `--static-evidence none|auto` to the primary CLI with default `none`. |
| SKE-CLI-02 | Add helper path functions for `<trace>.static-evidence.json` and `<trace>.static-evidence/`, falling back to staging paths without `--output`. |
| SKE-CLI-03 | Wrap static evidence collection/write in nonfatal helper behavior and preserve existing trace parsing/exit-code logic. |
| SKE-CLI-04 | Add a compact `summary` object to the sidecar payload covering status, artifacts, tool runs, metadata/disassembly presence, unsupported states, and claim boundaries. |

## Notes

- `ProblemPackager._is_cpp` already identifies HIP/C++ style compiled solutions.
- Phase 76 should write unsupported sidecars for non-HIP/C++ when `auto` is
  requested, rather than attempting Python/Triton/cache discovery.
- Direct helper tests can avoid running live GPU evaluation while still covering
  path naming, writing, unsupported outcomes, and summary shape.
