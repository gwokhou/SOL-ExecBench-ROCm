---
created: 2026-07-03T04:51:54.282Z
title: Canonicalize feedback sidecar identity fields
area: execbench
files:
  - src/sol_execbench/core/bench/agent_feedback.py
  - src/sol_execbench/core/bench/profile_summary.py
  - docs/agent_feedback_sidecar.md
  - docs/profile_summary_sidecar.md
  - tests/sol_execbench/test_agent_feedback.py
  - tests/sol_execbench/test_profile_summary.py
---

## Problem

SOL v1.36/v1.38 feedback sidecars support HIP-facing identity aliases, but the
contract should converge on one canonical identity vocabulary for future
freshness checks. HIP consumers should be able to prefer `target_id`, `run_id`,
`trace_path`, `candidate_id`, `source_sha256`, `sol_contract_version`, and
`sol_version` without guessing whether `candidate_hash`, `source_hash`, or
contract-version-only fields are the producer's primary identity.

Leaving the alias set open makes cross-repo freshness semantics harder to audit:
missing or mismatched identity fields could be interpreted differently by SOL
fixtures, HIP adapter normalization, and future release-gate diagnostics.

## Solution

Define the canonical identity fields in the SOL sidecar docs and schema tests.
Keep `candidate_hash` and `source_hash` as compatibility aliases for at least
one release, but make canonical fields the preferred producer output and fixture
shape. Add focused tests for canonical identity, legacy alias compatibility,
stale mismatch detection, and diagnostic-only governance. Coordinate with HIP so
its adapter keeps aliases only as compatibility fallbacks.
