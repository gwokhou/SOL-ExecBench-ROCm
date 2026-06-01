---
phase: 114-release-candidate-validation
status: passed
verified: 2026-06-01
requirements:
  - RCVAL-01
  - RCVAL-02
  - RCVAL-03
  - RCVAL-04
---

# Phase 114 Verification

## Result

`status: passed`

Phase 114 satisfies the roadmap goal: maintainers can run bounded
release-candidate validation, review recorded pass/fail summaries, include
optional ROCm/Docker/dataset-slice evidence, and classify non-passing outcomes
as blocking, deferred, or diagnostic-only with next actions.

## Requirement Coverage

| Requirement | Evidence | Status |
|-------------|----------|--------|
| RCVAL-01 | `scripts/release_candidate_validation.py` writes JSON/Markdown summaries; focused tests cover pass/fail CPU-safe classification. | passed |
| RCVAL-02 | Optional `--include-rocm-smoke` and `--include-docker-smoke` paths record evidence or unavailable/deferred rows. | passed |
| RCVAL-03 | `--include-dataset-slice` requires bounded `--dataset-limit`, records execution closure/trust paths, and guards against unbounded overrides. | passed |
| RCVAL-04 | Result records include status, classification, and `next_action`; docs define blocking/deferred/diagnostic-only policy. | passed |

## Verification Commands

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_release_candidate_validation.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_research_release_docs.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_release_candidate_validation.py tests/sol_execbench/test_run_docker_runtime_evidence.py -q
uv run --with ruff ruff check scripts/release_candidate_validation.py tests/sol_execbench/test_release_candidate_validation.py
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/release_candidate_validation.py --output-dir out/release_candidate_validation_phase114
```

## Human Verification

None required for Phase 114. Optional live ROCm, Docker, and dataset-slice runs
remain environment-dependent prerelease evidence paths.

## Residual Risk

- Optional live hardware evidence can still be unavailable on CPU-only hosts.
- The bounded dataset slice depends on local benchmark assets.
- This phase deliberately does not provide paper parity, upstream SOLAR parity,
  hosted leaderboard readiness, hard sandbox authority, CDNA4 validation, or
  MI300X/CDNA3 full-suite validation.
