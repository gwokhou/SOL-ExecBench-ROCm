# Phase 62 Plan

## Objectives

- Persist artifacts under a stable `<trace>.rocprofv3/` directory when trace
  output is requested.
- Register profiler artifacts in structured sidecar metadata.
- Keep profiler failures nonfatal by falling back to normal evaluation.

## Tasks

- [x] Add artifact discovery and classification.
- [x] Add profile sidecar path and writer.
- [x] Record profiler return code, stdout/stderr tails, skipped reason, failed
  reason, and artifact list.

## Verification

- Unit tests cover success, unavailable, failed, and disabled profile metadata.
- Public contract guardrails confirm trace JSONL top-level keys remain exact.
