---
status: complete
completed_at: "2026-07-07T12:12:51Z"
---

# Quick Task Summary

Refactored repeated helper implementations to shared utilities and standard
library/Pydantic APIs.

Changes:
- Added deterministic model JSON/checksum, JSON object loading, and JSONL
  Pydantic model loading helpers.
- Added shared text helpers for Markdown table cells, subprocess output text,
  tail extraction, and ordered uniqueness.
- Added shared scoring parsing helpers for strict JSON sidecar payloads.
- Replaced repeated report `to_json` and checksum code, Markdown cell escaping,
  selected JSON/JSONL model loading, ordered uniqueness, sha256 file hashing,
  and subprocess text/tail helpers.

Verification:
- `uv run pytest tests/sol_execbench/core/data/test_json_utils.py tests/sol_execbench/core/test_text_utils.py tests/sol_execbench/core/scoring/test_parsing_utils.py tests/sol_execbench/test_cli_problem_io.py tests/sol_execbench/test_trust_summary.py tests/sol_execbench/test_consistency_report.py tests/sol_execbench/test_evaluation_stability.py tests/sol_execbench/test_claim_upgrade.py tests/sol_execbench/test_paper_denominator_report.py tests/sol_execbench/test_amd_bound_sanity.py tests/sol_execbench/test_dataset_inventory_readiness.py tests/sol_execbench/test_dataset_migration.py tests/sol_execbench/test_dataset_run_closure.py tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_amd_sol_v2.py tests/sol_execbench/test_run_dataset_amd_score.py`
- `uv run pytest tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_environment_snapshot.py tests/sol_execbench/test_rocm_profiler.py tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/driver/test_problem_packager.py`
- `uv run --with ruff ruff check src/sol_execbench/core/data/json_utils.py src/sol_execbench/core/text_utils.py src/sol_execbench/core/scoring/parsing_utils.py src/sol_execbench/core src/sol_execbench/cli/problem_io.py src/sol_execbench/driver/problem_packager.py scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py scripts/internal/rdna4/run_rdna4_profiler_timing_smoke.py scripts/internal/release/build_prerelease_artifact_bundle.py scripts/internal/release/check_prerelease_readiness.py tests/sol_execbench/core/data/test_json_utils.py tests/sol_execbench/core/test_text_utils.py tests/sol_execbench/core/scoring/test_parsing_utils.py`
