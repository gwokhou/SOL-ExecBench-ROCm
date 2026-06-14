<!-- generated-by: gsd-doc-writer -->
# Development

SOL ExecBench ROCm Port is a Python package managed with `uv`. The package
source lives under `src/sol_execbench/`, the main console script is
`sol-execbench`, and the baseline comparison console script is
`sol-execbench-baseline`.

Development work should preserve the ROCm-only scope and the public benchmark
schemas unless a ROCm-specific change is required. Keep validation claims within
the evidence in the repository: current CDNA3 evidence is MI308X/`gfx942`
infrastructure evidence, not MI300X validation. MI300X and MI308X are sibling
CDNA3 GPU products that share `gfx942`; do not treat that as interchangeable
hardware-validation evidence. NVFP4/MXFP4 Quant ROCm adaptation and hardware
validation remain deferred until suitable CDNA4-class hardware is available.

## Local Setup

Install runtime and development dependencies with `uv`:

```bash
uv sync --all-groups
```

Run a GPU-free CLI contract check and the test suite to confirm the environment
is usable:

```bash
uv run sol-execbench contract --json
uv run pytest tests/
```

Install the local pre-commit hooks after dependencies are present:

```bash
uv run pre-commit install
```

The hook configuration runs Ruff check and format on pre-commit, checks for a
DCO sign-off line on commit messages, and runs `ty check` on pre-push.

Use the provided Docker workflow when host ROCm tooling is not already
configured:

```bash
./scripts/run_docker.sh --build
```

GPU evaluation requires ROCm-capable AMD hardware, device access to `/dev/kfd`
and `/dev/dri`, ROCm user-space tooling, and a ROCm PyTorch build.

## Common Commands

| Command | Purpose |
| --- | --- |
| `uv sync --all-groups` | Install runtime and development dependency groups. |
| `uv build` | Build package artifacts with Hatchling. |
| `uv run sol-execbench <problem_dir> --solution <solution-path>` | Run one benchmark problem. |
| `uv run sol-execbench --definition definition.json --workload workload.jsonl --solution solution.json` | Run one problem from explicit files. |
| `uv run sol-execbench contract --json` | Print the GPU-free evaluator compatibility contract. |
| `uv run sol-execbench doctor --json` | Print ROCm environment diagnostics. |
| `uv run sol-execbench toolchain --json` | Print ROCm evidence-tool routing diagnostics. |
| `uv run sol-execbench toolchain --json --list-registry` | Print registered ROCm toolchain capabilities. |
| `uv run sol-execbench dataset migrate-sol <source_root> <output_root>` | Convert locally downloaded SOL-ExecBench inputs into local benchmark layout. |
| `uv run sol-execbench dataset migrate-flashinfer <source_root> <output_root>` | Convert locally downloaded FlashInfer Trace inputs into local benchmark layout. |
| `uv run sol-execbench-baseline --candidate <file> --baseline <file>` | Compare trace JSONL files. |
| `uv run scripts/run_dataset.py <downloaded-benchmark-dir> --limit 5` | Run a small downloaded dataset batch. |
| `uv run python scripts/internal/rdna4/run_rdna4_profiler_timing_coverage.py` | Build RDNA4 profiler-backed timing coverage reports from timing sidecars. |
| `uv run python scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py --workload-sharded` | Profile missing RDNA4 workload slices independently and aggregate complete manifests. |
| `uv run python scripts/internal/rdna4/run_rdna4_profiler_partial_failures.py` | Classify partial RDNA4 profiler-backed targets by failure mode and closure decision. |
| `uv run python scripts/internal/rdna4/run_rdna4_profiler_sharded_closure.py` | Audit remaining partial or profiler-blocked targets for workload-sharded closure. |
| `uv run pytest tests/` | Run the full test suite. |
| `uv run --with ruff ruff check .` | Run lint checks when Ruff is not already installed in the environment. |
| `uv run --with ruff ruff format .` | Format Python files when Ruff is not already installed in the environment. |
| `uv run ty check` | Run type checks for `src` and `tests`. |
| `./scripts/run_docker.sh --build` | Build and enter the ROCm Docker environment. |

After `uv sync --all-groups`, `ruff` is already available from the development
dependency group, so `uv run ruff check .` and `uv run ruff format .` are also
valid.

## Package Layout

| Path | Role |
| --- | --- |
| `src/sol_execbench/cli/` | Click command implementations for evaluation, metadata, diagnostics, dataset migration, and baseline comparison. |
| `src/sol_execbench/core/data/` | Pydantic v2 schemas for definitions, workloads, solutions, traces, shapes, dtypes, and the evaluator contract. |
| `src/sol_execbench/core/bench/` | Correctness, timing, clock locking, reward-hack checks, profiler integration, static evidence, IO, and runtime helpers. |
| `src/sol_execbench/core/dataset/` | Dataset layout, migration, inventory, readiness, manifests, checksums, closure, sharding, parity-gap, and low-precision compatibility helpers. |
| `src/sol_execbench/core/scoring/` | AMD score, AMD bound estimates and graphs, hardware models, SOL derivation helpers, and baseline artifacts. |
| `src/sol_execbench/core/` | Shared reporting, compatibility, diagnostics, toolchain routing, Docker matrix, runtime evidence, consistency, stability, trust summary, and claim-upgrade utilities. |
| `src/sol_execbench/driver/` | Problem staging plus generated HIP/C++ build and evaluation driver templates. |
| `src/sol_execbench/data/` | Packaged AMD hardware model data. |
| `tests/sol_execbench/` | Package unit tests, integration tests, documentation guardrails, dataset tests, scoring tests, and ROCm marker tests. |
| `tests/examples/` | Example workflow and consistency tests. |
| `tests/docker/dependencies/` | ROCm container dependency and runtime readiness checks. |
| `examples/` | Runnable example problems and solution manifests. |
| `scripts/` | Dataset, release, reporting, validation, and Docker helper scripts. |
| `docker/` | ROCm container build files and target metadata. |

## Development Workflow

Keep changes narrow and match nearby code style. Prefer extending existing
helpers and schemas instead of introducing new parallel abstractions.

When changing evaluator behavior, check the CLI, driver templates, schema
models, and tests together. Evaluation stages submitted code through
`ProblemPackager`; HIP/C++ solutions build through
`src/sol_execbench/driver/templates/build_ext.py`, and Python, Triton, and
native solutions run through `src/sol_execbench/driver/templates/eval_driver.py`.

Dataset workflow changes usually involve both importable helpers under
`src/sol_execbench/core/dataset/` and `scripts/run_dataset.py`. Derived phases
can use parallel CPU/I/O workers through `--phase derived --jobs`. Trace
collection can use `--execution-mode pipeline` to overlap CPU preparation with
the serial GPU evaluator. Do not describe pipeline mode as parallel GPU
evaluation; the benchmark-critical GPU subprocess path remains single-GPU-job
ordered.

Optional evidence sidecars are diagnostic surfaces, not replacements for
canonical trace JSONL. No-trace diagnostics, environment snapshots,
`rocprofv3` artifacts, static-kernel evidence, AMD score outputs, paper
denominator reports, parity gaps, consistency reports, stability reports, and
trust summaries should remain separate from public benchmark schemas unless a
schema change is intentional and tested.

Recent milestones moved focused debt out of monolithic paths without changing
public benchmark schemas.

| Boundary | Helper modules | Still owned by orchestrator/template |
| --- | --- | --- |
| Dataset execution | `core.dataset.run_state`, `core.dataset.run_closure`, `core.dataset.evidence_refs`, `core.dataset.sharding` | `scripts/run_dataset.py` CLI parsing, ROCm GPU/profiler subprocess phases, derived-phase worker scheduling, trace-stage pipeline scheduling, and high-level loop flow. |
| RDNA4 profiler timing closure | `core.dataset.profiler_timing_coverage` | `scripts/internal/rdna4/run_rdna4_profiler_timing_batch.py` target selection, workload-slice staging, manifest import, profiler subprocess execution, and aggregate sidecar writing. |
| Eval driver runtime | `core.bench.eval_runtime` | `driver/templates/eval_driver.py` subprocess context, staged wiring, correctness/timing loop, and integration smoke behavior. |
| AMD bound analysis | `core.scoring.amd_bound_classification`, `core.scoring.amd_bound_estimate_families` | FX/AST graph extraction, family annotation, and formula bodies in existing scoring modules. |
| SOLAR derivation | `core.scoring.solar_derivation_status` | Sidecar dataclasses, parser validation, semantic group construction, and rendering. |
| Static evidence | `core.bench.static_kernel_status` | Artifact persistence, tool routing, bounded extractor execution, and sidecar model definitions. |

These boundaries are maintainability aids. They are not security isolation,
hardware validation, paper-scale parity evidence, or leaderboard authority.

Provenance guardrails live outside the evaluator path. `provenance.toml`
records source attribution classifications, and
`tests/sol_execbench/test_provenance_policy.py` verifies active SPDX header
expectations. Header cleanup is source maintenance, not benchmark behavior.

## Code Style

Ruff and Ty configuration live in `pyproject.toml`.

- Use Python `>=3.12,<3.14`.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes, Pydantic models, and enum classes.
- Use clear, descriptive `test_*` names.
- Keep focused fixes free of broad refactors.
- Do not commit local caches, build artifacts, downloaded datasets, or generated
  benchmark output.
- Keep source headers aligned with `provenance.toml` and `docs/provenance.md`
  when adding, moving, or substantially rewriting files.

Run lint, format, and type checks with:

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
```

Generated data, downloaded datasets, examples, build artifacts, caches, and
virtual environments are excluded from Ruff checks.

## Testing

Pytest is configured in `pyproject.toml` with `pytest-xdist` defaults:
`-n auto --dist loadgroup`.

Run the full suite:

```bash
uv run pytest tests/
```

Run focused checks:

```bash
uv run pytest tests/sol_execbench/test_e2e.py
uv run pytest tests/sol_execbench/core/data/
uv run pytest tests/sol_execbench/driver/
uv run pytest tests/examples/test_examples.py -k consistency
```

Run timing tests that are skipped by default:

```bash
uv run pytest tests -m timing_serial -n 0
```

Run ROCm hardware-sensitive tests only on a ROCm-capable host or container with
device passthrough:

```bash
uv run pytest tests -m requires_rocm -n 0
uv run pytest tests -m requires_rdna4 -n 0
uv run pytest tests -m requires_cdna3 -n 0
uv run pytest tests -m requires_rocm_dev -n 0
```

Run ROCm container dependency checks inside the Docker environment:

```bash
./scripts/run_docker.sh --build
./scripts/run_docker.sh -- uv run pytest tests/docker/dependencies/
```

Run focused RDNA4 profiler timing closure regressions:

```bash
uv run pytest \
  tests/sol_execbench/test_profiler_timing_coverage.py \
  tests/sol_execbench/test_rdna4_profiler_timing_batch.py \
  tests/sol_execbench/test_rdna4_profiler_partial_failures.py \
  tests/sol_execbench/test_rdna4_profiler_sharded_closure.py -q
```

Markers are registered in `pyproject.toml` and `tests/conftest.py`.

| Marker | Purpose |
| --- | --- |
| `cpp` | Tests that compile HIP/C++ extensions. |
| `timing_serial` | GPU timing tests skipped by default unless selected with `-m timing_serial`. |
| `requires_rocm` | Tests requiring a ROCm GPU visible through PyTorch. |
| `requires_rocm_dev` | Tests requiring ROCm native extension headers under `/opt/rocm`. |
| `requires_ck` | Tests requiring Composable Kernel headers. |
| `requires_rocwmma` | Tests requiring rocWMMA headers. |
| `requires_rdna4` | Tests requiring an AMD RDNA 4 GPU such as `gfx1200`. |
| `requires_cdna3` | Tests requiring an AMD CDNA 3 GPU such as `gfx942`. |
| `requires_cutile` | Legacy NVIDIA cuTile marker skipped in this ROCm-only port. |

On Linux, ROCm marker handling checks `/dev/kfd` and `/dev/dri` before probing
PyTorch. `requires_cutile` always skips because this is a ROCm-only port.

## Contribution Practices

Contributions should start from an approved GitHub issue. Keep each change tied
to one concern, update tests with behavior changes, and update documentation
when changing public commands, schemas, Docker targets, scoring, timing,
profiling, evidence semantics, ROCm support claims, or examples.

Commit titles use the project format:

```text
#<Issue Number> - <Commit Title>
```

Sign commits with DCO sign-off:

```bash
git commit -s -m "#123 - Fix trace parsing"
```

Pull requests should link the approved issue, describe behavior changes, and
list the tests, lint checks, type checks, Docker checks, or GPU checks that were
run. Document hardware-specific assumptions in tests or PR notes.

## CI Expectations

The GitHub Actions code-quality workflow runs on Python 3.12 and 3.13. It
installs locked dependencies, runs Ruff and Ty, runs the CPU-safe package test
subset, and runs example consistency checks. It intentionally avoids tests that
need live ROCm GPU execution.

The representative CI command sequence is:

```bash
uv sync --locked --all-groups --python <matrix-version>
uv run ruff check .
uv run ty check
uv run pytest tests/sol_execbench \
  --ignore=tests/sol_execbench/driver/test_eval_driver.py \
  --ignore=tests/sol_execbench/test_e2e.py
uv run pytest tests/examples/test_examples.py -k consistency
```
