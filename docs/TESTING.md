<!-- generated-by: gsd-doc-writer -->
# Testing

Pytest is the project test framework. `pyproject.toml` configures
`pytest-xdist` with `-n auto --dist loadgroup`, so full-suite runs execute in
parallel by default.

## Test Framework and Setup

The development dependency group declares `pytest>=9.0.2`
and `pytest-xdist>=3.5` for parallel execution. `pyproject.toml`
sets the default pytest options to `-n auto --dist loadgroup`, so tests run
across available workers unless a command overrides xdist with `-n 0`.

Install development dependencies:

```bash
uv sync --all-groups
```

The development group includes `pytest`, `pytest-xdist`, `ruff`, and `ty`.

## Running Tests

Run the full suite:

```bash
uv run pytest tests/
```

Run a focused package test:

```bash
uv run pytest tests/sol_execbench/test_e2e.py
```

Run schema tests:

```bash
uv run pytest tests/sol_execbench/core/data/
```

Run driver tests:

```bash
uv run pytest tests/sol_execbench/driver/
```

Run example consistency tests:

```bash
uv run pytest tests/examples/test_examples.py -k consistency
```

Run timing tests that are skipped by default:

```bash
uv run pytest tests -m timing_serial -n 0
```

Run Docker dependency checks inside the ROCm container:

```bash
uv run pytest tests/docker/dependencies/
```

Run recent evaluation and dataset trustworthiness regressions:

```bash
uv run pytest \
  tests/sol_execbench/test_cli_environment_snapshot.py \
  tests/sol_execbench/core/bench/test_eval_runtime.py \
  tests/sol_execbench/core/data/test_solution.py \
  tests/sol_execbench/test_dataset_run_closure.py \
  tests/sol_execbench/test_run_dataset_execution_closure.py \
  tests/sol_execbench/test_dataset_failure_mode_docs.py \
  tests/sol_execbench/test_dataset_migration.py \
  tests/sol_execbench/test_dataset_inventory_readiness.py \
  tests/sol_execbench/test_low_precision_compatibility.py \
  tests/sol_execbench/test_dataset_redistribution_policy.py \
  tests/sol_execbench/test_dataset_sharding.py -q
```

Run focused RDNA4 profiler-backed timing closure regressions:

```bash
uv run pytest \
  tests/sol_execbench/test_profiler_timing_coverage.py \
  tests/sol_execbench/test_rdna4_profiler_timing_batch.py \
  tests/sol_execbench/test_rdna4_profiler_partial_failures.py \
  tests/sol_execbench/test_rdna4_profiler_sharded_closure.py -q
```

## Markers

Core markers are registered in `pyproject.toml`; environment-sensitive skip
logic and additional markers are registered in `tests/conftest.py`.

| Marker | Meaning |
| --- | --- |
| `cpp` | Compiles HIP/C++ extensions and may be slow. |
| `timing_serial` | GPU timing tests skipped by default unless selected with `-m timing_serial`. |
| `requires_rocm` | Requires a ROCm GPU visible through PyTorch. |
| `requires_rocm_dev` | Requires ROCm native extension development headers under `/opt/rocm`. |
| `requires_ck` | Requires Composable Kernel headers under `/opt/rocm/include/ck/`. |
| `requires_rocwmma` | Requires rocWMMA headers under `/opt/rocm/include/rocwmma/`. |
| `requires_rdna4` | Requires an AMD RDNA 4 GPU such as `gfx1200`. |
| `requires_cdna3` | Requires an AMD CDNA 3 GPU such as `gfx942`. |
| `requires_cutile` | Legacy NVIDIA cuTile marker skipped in this ROCm-only port. |

On Linux, `requires_rocm` collection first checks for `/dev/kfd` and
`/dev/dri`. If those device nodes are not visible, the test is skipped with an
explicit diagnostic before importing or probing PyTorch ROCm.
`requires_rdna4` accepts detected `gfx12*` devices and `requires_cdna3` accepts
detected `gfx94*` devices. There is no `requires_mi300x` or `requires_cdna4`
shortcut marker; CDNA3-family and CDNA4 validation claims require the evidence
listed in the claim and release guardrail docs.

## Hardware-Sensitive Tests

Run these only on a ROCm-capable Linux host or in a container with ROCm device
passthrough:

```bash
uv run pytest tests -m requires_rocm -n 0
uv run pytest tests -m requires_rdna4 -n 0
uv run pytest tests -m requires_cdna3 -n 0
uv run pytest tests -m requires_rocm_dev -n 0
```

For CDNA3 readiness specifically, the lightweight marker surface lives in
`tests/sol_execbench/test_cdna3_hardware_marker.py`:

```bash
uv run pytest tests/sol_execbench/test_cdna3_hardware_marker.py -m requires_cdna3 -n 0
```

On non-CDNA3 hosts this command should skip the live hardware test with a
reason such as `requires AMD CDNA 3 ROCm GPU` or the ROCm availability reason.
That skip is expected on RDNA 4 hosts, macOS hosts, and ROCm-less containers;
it is not a CDNA3 validation failure. Passing this marker test on `gfx94*`
hardware proves the test gate is usable; it is still not full MI300X hardware-validation evidence.
MI300X and MI308X are sibling GPU products under
the CDNA3 architecture family and share the `gfx942` code path, but recorded
MI308X/gfx942 validation-infrastructure evidence must not be reported as
MI300X validation.

For native library categories, use marker-filtered runs for the installed
headers:

```bash
uv run pytest tests -m requires_ck -n 0
uv run pytest tests -m requires_rocwmma -n 0
```

Docker dependency tests are hard readiness checks and should run only where ROCm
devices, user-space libraries, and container passthrough are expected:

```bash
./scripts/run_docker.sh --build
./scripts/run_docker.sh -- uv run pytest tests/docker/dependencies/
```

## ROCm Matrix Guardrails

The compatibility matrix and Docker target guardrails are designed to be
CPU-safe. They cover status classification, reason-code classification,
schema serialization, mixed-version blocking, claim flags, docs wording,
Docker Target selection, default behavior preservation, unknown Target rejection,
runtime evidence sidecars, dependency preflight behavior, dry-run wrapper
construction, and docs claim boundaries.

Focused matrix run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_rocm_compatibility_matrix.py \
  tests/sol_execbench/test_matrix_claim_guardrails.py \
  tests/sol_execbench/test_docker_matrix_targets.py \
  tests/sol_execbench/test_docker_matrix_preflight.py \
  tests/sol_execbench/test_run_docker_matrix_script.py \
  tests/sol_execbench/test_dependency_matrix_policy.py \
  tests/sol_execbench/test_dependency_matrix_classification.py \
  tests/sol_execbench/test_dependency_matrix_cli.py \
  tests/sol_execbench/test_run_docker_dependency_preflight.py \
  tests/sol_execbench/test_runtime_evidence_reports.py \
  tests/sol_execbench/test_run_docker_runtime_evidence.py \
  tests/sol_execbench/test_rocm_matrix_docs.py -q
```

Wrapper syntax and docs lint checks:

```bash
bash -n scripts/run_docker.sh
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  docs/CLAIMS.md docs/TESTING.md tests/sol_execbench/test_rocm_matrix_docs.py
```

## v1.19 Evidence Docs Guardrails

The centralized v1.19 guide is `docs/v1_19_evidence_guide.md`. Its focused
CPU-safe checks cover execution closure, paper denominator reports, Matrix
schema export, Matrix semantic diff, AMD bound sanity, and public wording
boundaries. v1.19 documentation has no full 235-problem paper validation, no
upstream SOLAR parity, no score authority, no leaderboard readiness, no full
MI300X validation under CDNA3, no CDNA4 validation, no native-host ROCm Matrix
validation, and no new-hardware validation.

Run the v1.19 documentation and sidecar-only contract checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_research_release_docs.py \
  tests/sol_execbench/test_rocm_matrix_docs.py \
  tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_paper_denominator_fields_remain_sidecar_only \
  tests/sol_execbench/test_public_contract_guardrails.py::test_v1_19_amd_bound_sanity_fields_remain_sidecar_only -q

UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  tests/sol_execbench/test_research_release_docs.py
```

This command set does not run GPU probes, ROCm live validation, Docker builds,
Docker containers, hardware-marker tests, dependency installs, or dependency
relocking.

## v1.20 Evidence Quality Docs Guardrails

The centralized v1.20 guide is `docs/v1_20_evidence_quality_guide.md`. Its
focused CPU-safe checks cover consistency lint, evaluation stability,
claim-upgrade rejection, trust summary rendering, deterministic serialization,
example fixtures, and public contract boundaries. v1.20 documentation has no
full 235-problem paper validation, no CDNA3-family validation, including MI300X
(`gfx942`), CDNA4 validation, native-host Matrix authority, hosted leaderboard
readiness, upstream SOLAR parity, or new-hardware validation. Recorded
MI308X/gfx942 infrastructure evidence remains bounded to that sibling CDNA3
product and does not upgrade MI300X claims.

Run the v1.20 documentation and sidecar-only contract checks:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_v1_20_evidence_quality_docs.py \
  tests/sol_execbench/test_public_contract_guardrails.py -q
```

## Evaluation And Dataset Trustworthiness Guardrails

The recent CPU-safe guardrails cover:

- bounded no-trace diagnostic sidecars and canonical trace JSONL preservation;
- staged Python import identity collisions;
- native compile option rejection for host path injection, response files, and
  unsafe linker/runtime loader behavior;
- eval-driver trace emission and reward-hack helper boundaries;
- dataset reuse and stale-provenance decisions;
- selected ready subsets, missing traces, missing derived evidence sidecars,
  CLI timeout/nonzero/no-output closure states, and rerun behavior;
- file-backed bounded CLI logs, release/prerelease transcript redaction across
  chunk boundaries, temporary stream cleanup, and non-GPU derived-phase
  `--jobs` scheduling;
- local SOL-ExecBench and FlashInfer Trace migration manifests, source
  revisions, checksum refs, missing-input blockers, and redistribution policy;
- ROCm readiness classes, blocker reports, ready-subset denominators, and
  no-hardware-validation claim boundaries;
- CPU-safe NVFP4/MXFP4/E2M1 compatibility helpers with explicit unvalidated
  CDNA4 evidence markers and CDNA3 hardware-unsupported skip behavior;
- deterministic dataset sharding plan and merge semantics.

Focused run:

```bash
uv run pytest \
  tests/sol_execbench/test_cli_environment_snapshot.py \
  tests/sol_execbench/core/bench/test_eval_runtime.py \
  tests/sol_execbench/core/data/test_solution.py \
  tests/sol_execbench/driver/test_build_ext.py \
  tests/sol_execbench/test_dataset_run_closure.py \
  tests/sol_execbench/test_run_dataset_execution_closure.py \
  tests/sol_execbench/test_run_dataset_amd_score.py \
  tests/sol_execbench/test_release_candidate_validation.py \
  tests/sol_execbench/test_prerelease_artifact_bundle.py \
  tests/sol_execbench/test_prerelease_readiness.py \
  tests/sol_execbench/test_dataset_failure_mode_docs.py \
  tests/sol_execbench/test_dataset_migration.py \
  tests/sol_execbench/test_dataset_inventory_readiness.py \
  tests/sol_execbench/test_low_precision_compatibility.py \
  tests/sol_execbench/test_dataset_redistribution_policy.py \
  tests/sol_execbench/test_dataset_sharding.py -q
```

These tests do not prove live ROCm shard execution or full dataset validation.
They exercise policy, provenance, path, diagnostic, and closure behavior without
requiring GPU hardware.
They also do not validate NVFP4/MXFP4 Quant ROCm adaptation; that adaptation
and hardware validation remain deferred until suitable CDNA4-class hardware is
available.

## Provenance And Prerelease Guardrails

Recent CPU-safe release checks cover source attribution policy, prerelease
artifact readiness, public-prerelease wording, and research-preview claim
boundaries:

```bash
uv run pytest \
  tests/sol_execbench/test_provenance_policy.py \
  tests/sol_execbench/test_prerelease_readiness.py \
  tests/sol_execbench/test_public_prerelease_docs.py \
  tests/sol_execbench/test_research_preview_docs.py -q
```

`tests/sol_execbench/test_provenance_policy.py` checks that files retaining
NVIDIA SPDX notices are listed under `provenance.toml` and that cleanup
candidates carry project attribution only. The prerelease readiness tests check
required bundle artifacts, checksum behavior, forbidden claim boundaries,
known-gap statuses, public MI300X-under-CDNA3 wording, unavailable CDNA4 wording,
and the provenance policy gate.

## Live ROCm Validation

Live ROCm validation is marker-gated. Use these checks only on a ROCm-capable
Linux host or container with ROCm device passthrough:

```bash
uv run pytest tests -m requires_rocm -n 0
uv run pytest tests -m requires_rdna4 -n 0
uv run pytest tests -m requires_cdna3 -n 0
```

CDNA3-only marker readiness can also be checked directly:

```bash
uv run pytest tests/sol_execbench/test_cdna3_hardware_marker.py -m requires_cdna3 -n 0
```

Interpret CDNA3 skips carefully: skipped tests on RDNA 4, macOS, or ROCm-less
containers mean the host is not a `gfx94*` validation target. They do not
upgrade or invalidate the deferred MI300X full-suite status. Passing CDNA3
marker-gated tests on MI308X/gfx942 is validation-infrastructure evidence for
that recorded sibling product, not a full MI300X hardware-validation pass.

For Matrix evidence, the current host ROCm 7.1.x environment may be recorded as
observed evidence through compatibility sidecars. ROCm 7.0.x or
ROCm 7.2.x native-host validation requires a matching host or separate machine;
default validation does not require host reinstall for ROCm 7.0.x or ROCm 7.2.x.
Docker rows for those Targets remain container ROCm user-space evidence on the
recorded host driver/devices unless direct native-host evidence is archived.

### Compatibility Matrix Summary

For day-to-day testing, treat the matrix as validation context rather than a
test-selection guide. The actionable commands are the marker-filtered pytest
runs above and the Docker wrapper checks in the preceding sections.

Recorded container rows currently cover the declared ROCm 7.0.2, 7.1.1, and
7.2.0 Docker targets on the recorded RDNA 4 host driver/devices. Those rows are
container ROCm user-space evidence only. They do not upgrade any target to
native-host validation, paper parity, score authority, or leaderboard
readiness.

| Target id | Local image tag | Requested ROCm user-space | Evidence summary |
| --- | --- | --- | --- |
| `rocm-7.0.2-ubuntu-24.04-container` | `sol-execbench:rocm-7.0.2-complete` | 7.0.2 | `linear_backward` passed 3/3 workloads with `--record-container-validation`; trace `rocm-7.0.2-linear-wrapper-official.jsonl` and sidecar `rocm-7.0.2-linear-wrapper-official.compatibility.json` record container_validated evidence, but `CLOCKS_LOCKED=0` leaves performance unlocked. |
| `rocm-7.1.1-ubuntu-24.04-container` | `sol-execbench:rocm-7.1.1-complete` | 7.1.1 | Default target with project-default target-specific PyTorch ROCm dependencies and `CLOCKS_LOCKED=1` when the recorded container-validation path succeeds. |
| `rocm-7.2.0-ubuntu-24.04-container` | `sol-execbench:rocm-7.2-complete` | 7.2.0 | `linear_backward` passed 3/3 workloads with trace `rocm-7.2-linear-wrapper-official.jsonl` and sidecar `rocm-7.2-linear-wrapper-official.compatibility.json`; `CLOCKS_LOCKED=1` was recorded for official wrapper evidence. |

Key interpretation points:

- `container_validated` means the selected container target ran through the
  wrapper path on recorded host devices and wrote compatibility sidecars.
- ROCm 7.0.2 evidence remains unlocked performance evidence because the clock
  lock command failed and the run recorded `CLOCKS_LOCKED=0`.
- ROCm 7.1.1 and 7.2.0 container rows recorded `CLOCKS_LOCKED=1`.
- Smoke runs through `--allow-untested-target-smoke` or
  `--allow-mixed-version-dependencies` are diagnostic and non-authoritative.
- Mixed-version diagnostics can report `benchmark_allowed=false` with
  `status=mixed_version`; those runs should not be upgraded into validation
  claims.
- ROCm 7.0 target-specific PyTorch ROCm smoke coverage uses
  `torch==2.10.0+rocm7.0`; ROCm 7.2 target-specific PyTorch ROCm smoke
  coverage uses `torch==2.11.0+rocm7.2`.
- Non-authoritative ROCm 7.2 smoke artifacts may be named
  `rocm-7.2-linear-wrapper-smoke.jsonl` and
  `rocm-7.2-linear-wrapper-smoke.compatibility.json`.
- Native-host validation requires direct native-host evidence for that ROCm
  stack; it cannot be inferred from Docker image selection or container runs.

Detailed historical E2E artifacts live under `.artifacts/e2e-*` when present in
a checkout or release bundle. Keep new historical run logs out of this testing
guide unless they change the commands developers should run.

## Writing New Tests

| Area | Typical Coverage |
| --- | --- |
| `tests/sol_execbench/core/data/` | Pydantic schemas and JSON model behavior. |
| `tests/sol_execbench/core/bench/` | Correctness, timing, clock locking, reward-hack, and utility behavior. |
| `tests/sol_execbench/driver/` | Staging, build template, and evaluation driver behavior. |
| `tests/sol_execbench/test_*` | End-to-end, migration, public contract, scoring, environment, Docker, and docs guardrails. |
| `tests/examples/` | Example file consistency and runnable workflow coverage. |
| `tests/docker/dependencies/` | ROCm runtime, HIP, PyTorch ROCm, Triton ROCm, and library dependency checks. |

Use `test_*.py` file names and descriptive `test_*` function names such as
`test_rejects_invalid_solution_schema`. Place new tests next to related
coverage under `tests/sol_execbench/`, or under `tests/examples/` for example
workflow coverage.

Shared test helpers live in `tests/conftest.py`,
`tests/sol_execbench_type_helpers.py`, and
`tests/sol_execbench/solar_derivation_fixtures.py`. Use `tmp_cache_dir` from
`tests/conftest.py` when a test needs an isolated `SOLEXECBENCH_CACHE_PATH`,
and use the typed model constructors in `tests/sol_execbench_type_helpers.py`
for schema-heavy tests.

## Coverage Requirements

No coverage threshold is configured in `pyproject.toml`, and no coverage
configuration file is present.

| Type | Threshold |
| --- | --- |
| Lines | No threshold configured. |
| Branches | No threshold configured. |
| Functions | No threshold configured. |
| Statements | No threshold configured. |

## CI Integration

The `Python Quality` workflow runs on push and pull request for Python 3.12 and
3.13. The workflow is defined in `.github/workflows/code-quality.yml` and uses
the `python-tests` job:

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ty check
uv run pytest tests/sol_execbench \
  --ignore=tests/sol_execbench/driver/test_eval_driver.py \
  --ignore=tests/sol_execbench/test_e2e.py
uv run pytest tests/examples/test_examples.py -k consistency
```

The remote workflow excludes tests that need live ROCm GPU execution or Docker
runtime passthrough. Run those locally on suitable hardware before merging
hardware-sensitive changes.
