---
quick_id: 260709-my4
slug: v1-43-tag-release-release-tag
status: complete
completed_at: "2026-07-09T08:40:00Z"
---

# Summary

Updated the SOL runtime release constant to `v1.43` and synchronized the active
default release identity tests plus current HIP-facing agent-feedback and
profile-summary fixtures. Stale fixtures intentionally remain on `v1.42`.

## Verification

Passed:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_contract.py tests/sol_execbench/core/bench/test_agent_feedback.py tests/sol_execbench/core/bench/test_agent_feedback_models.py tests/sol_execbench/core/bench/test_agent_feedback_fixtures.py tests/sol_execbench/core/bench/test_profile_summary.py tests/sol_execbench/core/bench/test_profile_summary_fixtures.py tests/sol_execbench/cli/sidecars/test_profile.py -q
```

Result: `76 passed in 1.81s`.
