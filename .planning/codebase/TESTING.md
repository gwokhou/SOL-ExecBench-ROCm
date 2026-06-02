---
generated_at: 2026-06-02
last_mapped_commit: 8019adc6295a78d4636037889245abcb3f9a52bb
focus: quality
---

# Testing

## Framework

- Tests use pytest.
- Parallel execution is enabled by default through `addopts = "-n auto --dist loadgroup"` in `pyproject.toml`.
- CI runs Python 3.12 and 3.13 in `.github/workflows/code-quality.yml`.
- CI checks `uv sync --locked --all-groups`, `ruff check`, `ty check`, a CPU-safe `tests/sol_execbench` subset, and `tests/examples/test_examples.py -k consistency`.

## Markers

- `requires_rocm`: requires a ROCm GPU visible through PyTorch.
- `requires_rocm_dev`: requires ROCm native extension development headers.
- `requires_rdna4`: requires an AMD RDNA 4 GPU such as `gfx1200`.
- `requires_cdna3`: requires an AMD CDNA 3 GPU such as `gfx942`.
- `requires_ck`: requires Composable Kernel headers.
- `requires_rocwmma`: requires rocWMMA headers.
- `requires_cutile`: legacy NVIDIA cuTile marker; always skipped in this ROCm-only port.
- `timing_serial`: GPU timing tests skipped unless explicitly selected with `-m timing_serial`.

## Test Organization

- Schema tests live under `tests/sol_execbench/core/data/`.
- Benchmark helper tests live under `tests/sol_execbench/core/bench/`.
- Driver and staging tests live under `tests/sol_execbench/driver/`.
- Dataset, scoring, evidence, release, and docs guardrails live in top-level `tests/sol_execbench/test_*.py`.
- Docker dependency checks live under `tests/docker/dependencies/`.
- Public examples are covered by `tests/examples/`.

## Coverage Themes

- Schema and public contract guardrails, including CUDA/NVIDIA value rejection.
- Correctness, dtype, shape, timing, clock, and output allocation behavior.
- Reward-hack defenses for monkey patching, hidden streams, lazy outputs, thread injection, subclasses, and dynamic extension loading.
- Dataset execution closure, ready subsets, denominator accounting, sharding, parity gaps, and run state.
- AMD-native scoring, SOL/SOLAR-derived evidence, bound estimates, static kernel evidence, consistency, stability, trust summaries, and claim-upgrade logic.
- Release readiness, prerelease artifact bundles, provenance policy, public prerelease docs, and research preview docs.

## Hardware-Sensitive Tests

- Hardware tests are skipped automatically when `/dev/kfd`, `/dev/dri`, PyTorch ROCm, development headers, or architecture-specific GPUs are unavailable.
- RDNA 4 has recorded validation evidence in project docs.
- Full MI300X validation on CDNA3 remains deferred until a complete real-hardware evidence chain exists.
- CDNA4 validation is unavailable because suitable hardware is not currently accessible.

## Common Commands

- Full suite: `uv run pytest tests/`.
- CPU-safe package tests: `uv run pytest tests/sol_execbench --ignore=tests/sol_execbench/driver/test_eval_driver.py --ignore=tests/sol_execbench/test_e2e.py`.
- Focused driver tests: `uv run pytest tests/sol_execbench/driver/test_eval_driver.py -q`.
- ROCm-marked tests: `uv run pytest -m requires_rocm -q -rs`.
- Docker dependency checks: `uv run pytest tests/docker/dependencies -q`.
- Lint/type checks: `uv run ruff check .` and `uv run ty check`.

## Test Risks

- CI excludes GPU e2e and generated driver tests, so local ROCm verification remains important for release claims.
- Timing tests require serial execution and stable clocks for meaningful performance assertions.
- Docker rows are container user-space evidence and must not be treated as native-host validation.
