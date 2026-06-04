---
status: passed
phase: 131
phase_name: Dataset License and Provenance Policy
verified_at: 2026-06-04
---

# Phase 131 Verification

## Result

Passed.

## Must-Haves

- Machine-readable provenance distinguishes NVIDIA SOL-ExecBench, FlashInfer
  Trace, generated local artifacts, and project-owned ROCm code.
  - Covered by `provenance.toml` dataset policy and
    `test_provenance_manifest_defines_dataset_redistribution_policy`.
- Redistribution policy classifies publishable, local-only, generated-only,
  excluded, and release-bundle-blocked boundaries.
  - Covered by `provenance.toml` and policy tests.
- CPU-safe guardrails fail if restricted NVIDIA original or derivative dataset
  content is staged or included in release bundles.
  - Covered by `scripts/check_dataset_redistribution.py` and prerelease
    readiness tests.
- Documentation preserves NVIDIA Evaluation Dataset License and Apache-2.0
  FlashInfer Trace attribution boundaries.
  - Covered by docs updates and provenance/prerelease tests.

## Commands

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_provenance_policy.py tests/sol_execbench/test_dataset_redistribution_policy.py tests/sol_execbench/test_prerelease_readiness.py
UV_CACHE_DIR=/tmp/uv-cache uv run --with ruff ruff check scripts/check_dataset_redistribution.py scripts/check_prerelease_readiness.py tests/sol_execbench/test_provenance_policy.py tests/sol_execbench/test_dataset_redistribution_policy.py tests/sol_execbench/test_prerelease_readiness.py
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/check_dataset_redistribution.py --path data/SOL-ExecBench/benchmark/L1/problem/definition.json --json
```

## Results

- Pytest: `18 passed`
- Ruff: `All checks passed!`
- Guardrail smoke: expected blocking exit for synthetic
  `data/SOL-ExecBench/benchmark/L1/problem/definition.json`, with
  `source_id: nvidia_sol_execbench`
