---
generated_by: gsd-map-codebase
focus: quality
mapped_at: 2026-06-16
---

# Testing

## Framework

- Pytest is the test framework.
- `pytest-xdist` is enabled by default through `pyproject.toml` with
  `-n 8 --dist loadgroup`.
- The worker count is intentionally capped because each worker can import
  PyTorch and ROCm libraries.
- Development dependencies include `pytest>=9.0.2` and `pytest-xdist>=3.5`.

## Test Locations

- Main package tests: `tests/sol_execbench/`.
- Driver tests: `tests/sol_execbench/driver/`.
- Example workflow tests: `tests/examples/`.
- Docker dependency probes: `tests/docker/dependencies/`.
- Shared test helpers: `tests/conftest.py` and
  `tests/sol_execbench_type_helpers.py`.
- Sample benchmark fixtures: `tests/sol_execbench/samples/`.
- Reward-hack manifests: `tests/samples/rmsnorm/`.

## Markers And Skip Logic

Markers are registered in `pyproject.toml` and `tests/conftest.py`:

- `cpp`
- `timing_serial`
- `requires_rocm`
- `requires_rocm_dev`
- `requires_ck`
- `requires_rocwmma`
- `requires_rdna4`
- `requires_cdna3`
- `requires_cutile`

`tests/conftest.py` checks `/dev/kfd`, `/dev/dri`, PyTorch ROCm availability,
detected `gfx*` architecture, ROCm development headers, CK headers, and rocWMMA
headers before hardware-sensitive tests run.

## Common Test Commands

- Full suite: `uv run pytest tests/`.
- Focused E2E: `uv run pytest tests/sol_execbench/test_e2e.py`.
- Driver tests: `uv run pytest tests/sol_execbench/driver/`.
- Example consistency: `uv run pytest tests/examples/test_examples.py -k consistency`.
- Timing tests: `uv run pytest tests -m timing_serial -n 0`.
- ROCm hardware tests: `uv run pytest tests -m requires_rocm -n 0`.
- RDNA4 tests: `uv run pytest tests -m requires_rdna4 -n 0`.
- CDNA3 tests: `uv run pytest tests -m requires_cdna3 -n 0`.
- Docker dependency checks inside container:
  `./scripts/run_docker.sh -- uv run pytest tests/docker/dependencies/`.

## CI Coverage

`.github/workflows/code-quality.yml` runs:

1. `uv sync --locked --all-groups --python <matrix-version>`
2. `uv run ruff check .`
3. `uv run ty check`
4. CPU-safe package tests under `tests/sol_execbench`, excluding the eval-driver
   and E2E files.
5. Example consistency tests with `tests/examples/test_examples.py -k consistency`.

The matrix covers Python 3.12 and 3.13 on Ubuntu.

## Test Themes

- Schema contracts and migration behavior.
- CLI behavior and environment snapshots.
- Driver staging and generated eval-driver runtime.
- Correctness, timing, reward-hack, profiler, static-evidence, and sidecar
  behavior.
- Dataset migration, inventory, readiness, closure, denominator, sharding, and
  redistribution boundaries.
- AMD scoring, AMD bound estimates, SOLAR derivation, and claim guardrails.
- ROCm matrix, Docker target, dependency, runtime evidence, and release docs.

## Practical Notes

- Hardware tests should generally run with `-n 0` to avoid multiple workers
  competing for one GPU.
- CPU-safe tests may model ROCm states but must not claim live hardware
  validation.
- Tests that assert documentation wording are part of the claim-boundary
  safety net.
