---
quick_id: 260616-mpd
status: complete
created: 2026-06-16
completed: 2026-06-16
---

# Quick Task 260616-mpd: Close HIP v1.26 SOL Feedback Sidecar Quick Gaps

## Goal

Close the SOL-side ambiguities found when comparing HIP Playground v1.26 plans
against SOL ExecBench ROCm v1.36.

## Must Haves

1. Clarify `profile_summary.sidecar.v1` semantics so HIP knows the current
   profile metadata sidecar remains `<trace>.profile.json` and that the
   profile-summary token is reserved/optional unless a future schema is added.
2. Populate generated agent-feedback identity with deterministic run,
   candidate, and source hashes when available from the emitted trace output.
3. Restrict SOL-emitted `items[].bottleneck` to a closed schema vocabulary with
   `unknown` fallback.

## Tasks

| Task | Files | Verification |
|------|-------|--------------|
| 1. Documentation clarification | `docs/user/EVALUATOR-CONTRACT.md`, `docs/user/agent_feedback_sidecar.md` | docs tests / grep |
| 2. Identity derivation | `src/sol_execbench/core/bench/agent_feedback.py`, `src/sol_execbench/cli/main.py` | pytest sidecar tests |
| 3. Bottleneck vocabulary | `src/sol_execbench/core/bench/agent_feedback.py`, fixtures/tests | pytest schema tests |

## Verification Commands

```bash
uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_agent_feedback_fixtures.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_v1_20_evidence_quality_docs.py
uv run --with ruff ruff check src/sol_execbench/core/bench/agent_feedback.py src/sol_execbench/cli/main.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_agent_feedback_fixtures.py tests/sol_execbench/test_cli_environment_snapshot.py
```
