---
status: complete
completed: 2026-07-03
quick_id: 260703-gze
slug: close-the-sol-side-diagnostic-only-feedb
---

# SOL Feedback Loop Release Gap Summary

Completed the SOL-side diagnostic-only feedback-loop release gap closure for
hip-playground.

Changes:
- Verified the current evaluator contract advertises
  `agent_feedback.sidecar.v1` and `profile_summary.sidecar.v1`.
- Verified the CLI evaluation path writes canonical Trace JSONL first, then
  writes `<trace>.profile-summary.json` and `<trace>.agent-feedback.json` next
  to that trace output.
- Added preferred HIP-facing freshness aliases:
  `sol_version`, `candidate_id`, and `source_sha256`.
- Preserved and validated compatibility aliases:
  `sol_contract_version`, `candidate_hash`, and `source_hash`.
- Kept profiler-derived `summary.bottleneck_hints[]` in
  `profile_summary.sidecar.v1` and documented that SOL does not duplicate them
  into `agent_feedback.items[]`.
- Added `docs/releases/v1_38_feedback_loop_rc1.md` as the prepared release tag
  candidate note for `v1.38-feedback-loop-rc1`.

Verification:
- `uv run pytest tests/sol_execbench/test_contract.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_profile_summary.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_agent_feedback_fixtures.py tests/sol_execbench/test_profile_summary_fixtures.py tests/sol_execbench/test_feedback_loop_release_candidate.py -q`
  - 77 passed
- `uv run --with ruff ruff check src/sol_execbench/core/bench/agent_feedback.py src/sol_execbench/core/bench/profile_summary.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_profile_summary.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_agent_feedback_fixtures.py tests/sol_execbench/test_profile_summary_fixtures.py tests/sol_execbench/test_feedback_loop_release_candidate.py`
  - All checks passed
