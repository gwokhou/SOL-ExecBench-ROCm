---
quick_id: 260616-mpd
status: complete
completed: 2026-06-16
---

# Quick Task 260616-mpd Summary: Close HIP v1.26 SOL Feedback Sidecar Quick Gaps

## Result

Closed the SOL-side quick gaps identified during HIP Playground v1.26
compatibility review:

- Clarified `profile_summary.sidecar.v1` as a reserved optional capability and
  documented that current ROCm profiler metadata remains `<trace>.profile.json`.
- Added deterministic CLI-generated feedback identity fields from emitted trace
  data: `target_id`, `run_id`, and `candidate_hash`.
- Kept `source_hash` empty unless actual source contents are available to the
  producer, avoiding a misleading trace-label hash.
- Restricted SOL-produced `items[].bottleneck` values to a closed schema
  vocabulary with `unknown` as the safe fallback label.

## Verification

```bash
uv run pytest tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_agent_feedback_fixtures.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_v1_20_evidence_quality_docs.py
uv run --with ruff ruff check src/sol_execbench/core/bench/agent_feedback.py src/sol_execbench/cli/main.py tests/sol_execbench/test_agent_feedback.py tests/sol_execbench/test_agent_feedback_fixtures.py tests/sol_execbench/test_cli_environment_snapshot.py
```

Both commands passed.
