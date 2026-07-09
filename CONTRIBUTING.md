<!-- generated-by: gsd-doc-writer -->
# Contributing

Thank you for contributing to SOL ExecBench ROCm Port. This repository is a
ROCm-only Python port of SOL ExecBench, so changes should preserve public
benchmark schemas and evaluation semantics unless a ROCm-specific difference is
required.

## Start From an Issue

Contributions should start from an approved GitHub issue. Keep each pull
request focused on one concern and link the issue in the PR description.

## Development Setup

Install dependencies:

```bash
uv sync --all-groups
```

Run core checks:

```bash
uv run ruff check .
uv run ty check
uv run pytest tests/
```

The CI workflow uses the locked dependency set and a CPU-safe test subset:

```bash
uv sync --locked --all-groups --python <3.12-or-3.13>
uv run ruff check .
uv run ty check
uv run pytest tests/sol_execbench \
  --ignore=tests/sol_execbench/driver/test_eval_driver.py \
  --ignore=tests/sol_execbench/test_e2e.py
uv run pytest tests/examples/test_examples.py -k consistency
```

For ROCm GPU evaluation, use a ROCm-capable AMD host or the Docker helper:

```bash
./scripts/run_docker.sh --build
```

See [Getting Started](docs/GETTING-STARTED.md), [Development](docs/DEVELOPMENT.md),
and [Testing](docs/TESTING.md) for more detail.

## Coding Standards

- Use Python `>=3.12,<3.14`; `.python-version` pins local development to
  Python 3.12.
- Follow Ruff formatting and linting from `pyproject.toml`.
- Run Ty type checks over `src` and `tests`.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes, Pydantic models, and enum classes.
- Keep changes consistent with nearby code.
- Avoid broad refactors in focused fixes.
- Do not commit local caches, virtual environments, build artifacts, downloaded
  datasets, generated benchmark output, credentials, or proprietary kernels.

Format and lint:

```bash
uv run ruff format .
uv run ruff check .
```

Type check:

```bash
uv run ty check
```

The `Python Quality` GitHub Actions workflow enforces Ruff, Ty, CPU-safe package
tests, and example consistency tests on Python 3.12 and 3.13.

Default pre-commit setup installs Ruff hooks, the commit-message DCO check,
and the pre-push Ty check:

```bash
uv run pre-commit install
```

## Tests

Place new package tests under `tests/sol_execbench/` near related coverage.
Place example workflow tests under `tests/examples/`.

Use environment-sensitive markers where appropriate. Base marker declarations
live in `pyproject.toml`; additional hardware and environment-sensitive marker
registration plus skip behavior live in `tests/conftest.py`.

- `cpp`
- `timing_serial`
- `requires_rocm`
- `requires_rocm_dev`
- `requires_ck`
- `requires_rocwmma`
- `requires_rdna4`
- `requires_cdna3`
- `requires_cutile` for legacy NVIDIA-only coverage that should be skipped in
  this ROCm-only port

`timing_serial` tests are skipped by default. Run them explicitly and disable
xdist for live timing checks:

```bash
uv run pytest tests -m timing_serial -n 0
```

Run ROCm, architecture, and native-library marker selections only on hosts or
containers with the matching devices and headers:

```bash
uv run pytest tests -m requires_rocm -n 0
uv run pytest tests -m requires_rdna4 -n 0
uv run pytest tests -m requires_cdna3 -n 0
uv run pytest tests -m requires_rocm_dev -n 0
uv run pytest tests -m requires_ck -n 0
uv run pytest tests -m requires_rocwmma -n 0
```

Place future CDNA3-specific live tests near related package coverage under
`tests/sol_execbench/` and mark them with both `requires_rocm` and
`requires_cdna3` when they need a real `gfx94*` GPU. Use
`tests/sol_execbench/core/platform/test_cdna3_hardware_marker.py` as the minimal marker-gate
pattern. CPU-safe tests may cover `gfx940`, `gfx941`, and `gfx942` schema or
metadata behavior, but those tests must not claim hardware validation.

For hardware-sensitive changes, record the ROCm version, GPU architecture, GPU
product, container target if used, exact test commands, ROCm timing evidence,
AMD-native score, and any NVFP4/MXFP4 deferred status in the PR.

Current CDNA3 evidence is MI308X (`gfx942`) validation infrastructure evidence,
not MI300X validation. MI300X and MI308X are sibling GPU products under the
CDNA3 architecture family and share `gfx942`, but do not describe MI308X runs
as MI300X hardware validation. For any CDNA3 validation change, record whether
the run produced the relevant evidence chain: full pytest log, dataset summary,
environment report, clock-lock evidence, per-problem traces, ROCm timing evidence,
AMD-native score report, FP8 status, and NVFP4/MXFP4 deferred status. Until that
evidence exists for the exact claim, describe CDNA3 work as schema support, test
readiness, infrastructure evidence, or deferred validation.

NVFP4/MXFP4 Quant ROCm adaptation and hardware validation are deferred until
CDNA4-class hardware is available. CDNA3 expected skips for those problems are
not CPU validation, dequantized benchmark validation, or performance evidence.

## Documentation

Update documentation when changing:

- Public CLI behavior
- Benchmark schemas
- Solution language categories
- Docker targets or runtime assumptions
- ROCm hardware support claims
- Scoring, timing, profiling, or evidence semantics
- Dataset reuse, execution-closure, failure-mode, or sharding semantics
- Dataset runner phase scheduling, bounded subprocess logging, release
  transcript handling, or derived evidence sidecar naming
- RDNA4 profiler-backed timing coverage, workload-sharded profiler manifests,
  partial failure classification, or sharded closure audit semantics
- Dataset migration, readiness classification, ready-subset denominator,
  low-precision compatibility, or redistribution-boundary behavior
- User-facing examples

Use conservative wording for unvalidated hardware or infrastructure claims.
Do not infer native-host validation from Docker/container evidence, MI300X
validation from MI308X evidence, CDNA4 validation without CDNA4-class hardware,
or NVFP4/MXFP4 validation from expected skips.

## Commit Messages

Use imperative commit titles in this project format:

```text
#<Issue Number> - <Commit Title>
```

Sign commits with DCO sign-off:

```bash
git commit -s -m "#123 - Fix trace parsing"
```

The repository pre-commit configuration includes a commit-message hook that
checks for a `Signed-off-by:` line.

## PR Guidelines

The default branch is `main`. No repository branch naming convention is
currently documented; use a focused branch name that describes the issue or
change.

PRs should include:

- Linked approved issue
- Clear summary of behavior changes
- Tests, lint, type checks, Docker checks, or GPU checks run
- Documentation updates for public behavior changes
- Hardware assumptions for ROCm-sensitive changes
- Notes about any remaining validation gap

No repository PR template is currently present under `.github/`, so include
these details directly in the pull request description.

The `Python Quality` GitHub Actions workflow runs `uv sync --locked
--all-groups --python <matrix-version>`, Ruff, Ty, CPU-safe package tests, and
example consistency tests on Python 3.12 and 3.13.

## Issue Reporting

Use the GitHub Issues page for this repository:
`https://github.com/gwokhou/SOL-ExecBench-ROCm/issues`.

When reporting a bug or compatibility gap, include:

- Command run
- Expected behavior
- Actual behavior
- Python version
- PyTorch and ROCm versions
- GPU architecture, such as `gfx1200` or `gfx942`
- Operating system and container target, if relevant
- Trace JSONL, sidecars, logs, or sample problem paths when safe to share

Do not include credentials, Hugging Face tokens, proprietary kernels, private
datasets, or other sensitive data.

No repository issue templates are currently present under `.github/`, so include
the relevant environment and reproduction details in the issue body.
