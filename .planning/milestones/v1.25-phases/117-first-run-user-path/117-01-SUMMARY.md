---
phase: 117
phase_name: "First-Run User Path"
plan: 1
status: complete
completed_at: "2026-06-01"
requirements: [FIRST-01, FIRST-02, FIRST-03, FIRST-04]
---

# Phase 117 Summary

## Completed Work

- Added a first-run checklist to `docs/GETTING-STARTED.md`.
- Documented a minimal sample command that writes
  `out/first-run.trace.jsonl`.
- Added a first-trace reading section covering status, correctness, latency,
  speedup, and environment fields.
- Expanded troubleshooting with no-trace diagnostics, sidecars, and known
  limitations.
- Clarified that `torch.cuda` is a PyTorch ROCm compatibility namespace, not
  NVIDIA CUDA runtime support.
- Added a CPU-safe docs guardrail in
  `tests/sol_execbench/test_research_release_docs.py`.

## Evidence

- Plan/context commit: `ec71aab docs(117): plan first-run user path`
- Implementation commit: `4dcb9b8 docs: clarify first-run ROCm user path`
- Review: `.planning/phases/117-first-run-user-path/117-REVIEW.md`

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_contract_guardrails.py -q
```

Result: `67 passed in 3.07s`.

## Requirement Closure

- FIRST-01: Complete. Install and minimal example commands are documented.
- FIRST-02: Complete. Trace JSONL generation and field interpretation are
  documented.
- FIRST-03: Complete. Doctor, sidecars, no-trace diagnostics, and known
  limitations are visible from the first-run guide.
- FIRST-04: Complete. CUDA-named PyTorch APIs are framed as ROCm compatibility
  names, not NVIDIA runtime support.
