<!-- generated-by: gsd-doc-writer -->
# Development

This project is a Python 3.12+ package managed with `uv`. Development should
preserve the ROCm-only scope and the public benchmark schemas unless a
ROCm-specific change is required.

## Local Setup

Fork the repository, clone your fork, and install all dependency groups with
`uv`:

```bash
git clone <your-fork-url>
cd SOL-ExecBench-ROCm
uv sync --all-groups
```

Run the GPU-free contract check and the full test suite to confirm the local
environment is usable:

```bash
uv run sol-execbench contract --json
uv run pytest tests/
```

Optional pre-commit setup:

```bash
uv run --with pre-commit pre-commit install
uv run --with pre-commit pre-commit install --hook-type commit-msg
```

Use the Docker environment for ROCm GPU evaluation when host tooling is not
already configured:

```bash
./scripts/run_docker.sh --build
```

## Build Commands

This is not a JavaScript project. Development commands are defined by
`pyproject.toml`, console script entry points, and repository helper scripts.

| Command | Description |
| --- | --- |
| `uv sync --all-groups` | Install runtime and development dependencies. |
| `uv build` | Build the Python package artifacts with Hatchling. |
| `uv run sol-execbench <problem_dir> --solution <solution-path>` | Run one benchmark problem. |
| `uv run sol-execbench contract --json` | Print GPU-free evaluator compatibility metadata. |
| `uv run sol-execbench doctor --json` | Print ROCm environment diagnostics. |
| `uv run sol-execbench toolchain --json` | Print ROCm evidence-tool routing. |
| `uv run sol-execbench-baseline --candidate <file> --baseline <file>` | Compare trace JSONL files. |
| `uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5` | Run a small dataset batch. |
| `uv run pytest tests/` | Run the full test suite. |
| `uv run ruff check .` | Run lint checks. |
| `uv run ruff format .` | Format Python files. |
| `uv run ty check` | Run type checks over `src` and `tests`. |
| `./scripts/run_docker.sh --build` | Build and enter the ROCm Docker environment. |

## Code Style

Ruff is configured in `pyproject.toml`, and Ty is configured with source roots
under `[tool.ty.src]`.

- Use Python `>=3.12,<3.14`.
- Use `snake_case` for modules, functions, and variables.
- Use `PascalCase` for classes, Pydantic models, and enum classes.
- Keep changes consistent with nearby modules.
- Avoid broad refactors in focused fixes.
- Do not commit local caches, build artifacts, downloaded datasets, or generated
  benchmark output.

Run style and type checks with:

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
```

The Ruff config excludes generated data, downloaded datasets, examples, build
artifacts, caches, and virtual environments.

## Source Areas

| Area | Purpose |
| --- | --- |
| `src/sol_execbench/cli/` | Click entry points for evaluation, metadata, diagnostics, routing, and baseline comparison. |
| `src/sol_execbench/core/data/` | Pydantic schemas for definitions, workloads, solutions, traces, shapes, dtypes, and contracts. |
| `src/sol_execbench/core/bench/` | Correctness, timing, clock locking, reward-hack guardrails, profiler integration, static evidence, and IO helpers. |
| `src/sol_execbench/core/dataset/` | Dataset layout, inventory, readiness, manifest, checksum, ready-subset, reuse, closure, sharding, and parity-gap helpers. |
| `src/sol_execbench/core/scoring/` | AMD scoring, bound estimates, bound graphs, hardware models, SOL derivation, and baseline artifacts. |
| `src/sol_execbench/driver/` | Problem staging and generated compile/evaluation templates. |
| `src/sol_execbench/data/` | Packaged AMD hardware model JSON. |
| `tests/sol_execbench/` | Package unit, integration, migration, scoring, and documentation guardrail tests. |
| `tests/examples/` | Example consistency and workflow coverage. |
| `tests/docker/dependencies/` | Container dependency and ROCm runtime readiness checks. |

## Helper Boundaries

Recent milestones moved focused debt out of monolithic paths without changing
public benchmark schemas.

| Boundary | Helper modules | Still owned by orchestrator/template |
| --- | --- | --- |
| Dataset execution | `core.dataset.run_state`, `core.dataset.run_closure`, `core.dataset.evidence_refs`, `core.dataset.sharding` | `scripts/run_dataset.py` CLI parsing, serial subprocess invocation, and high-level loop flow. |
| Eval driver runtime | `core.bench.eval_runtime` | `driver/templates/eval_driver.py` subprocess context, staged wiring, correctness/timing loop, and integration smoke behavior. |
| AMD bound analysis | `core.scoring.amd_bound_classification`, `core.scoring.amd_bound_estimate_families` | FX/AST graph extraction, family annotation, and formula bodies in existing scoring modules. |
| SOLAR derivation | `core.scoring.solar_derivation_status` | Sidecar dataclasses, parser validation, semantic group construction, and rendering. |
| Static evidence | `core.bench.static_kernel_status` | Artifact persistence, tool routing, bounded extractor execution, and sidecar model definitions. |

These boundaries are maintainability aids. They are not security isolation,
hardware validation, paper-scale parity evidence, or leaderboard authority.

Dataset helper additions include reuse policy decisions, stale-provenance
mismatch normalization, selected-workload closure record assembly, evidence gap
classification, and deterministic shard/merge semantics. The sharding helper is
an importable design path; it does not add dataset CLI parallelism by itself.

Eval-driver helper additions include strict trace JSONL emission and
reward-hack boundary helpers. Python submissions are loaded through unique
staged module identities to avoid collisions with already-imported modules.
Native compile options are validated before extension loading to reject host
path injection, response files, and unsafe runtime loader/linker behavior.

## CLI And Runtime Notes

The main console script is `sol-execbench`, backed by
`sol_execbench.cli:cli`. The baseline comparison script is
`sol-execbench-baseline`, backed by `sol_execbench.cli.baseline:cli`.

Normal evaluation stages files into a temporary directory through
`ProblemPackager`. HIP/C++ solutions compile with
`src/sol_execbench/driver/templates/build_ext.py` and then run through
`src/sol_execbench/driver/templates/eval_driver.py`; Python and Triton solutions
run directly through the evaluation driver. Subprocesses receive
`PYTORCH_ALLOC_CONF=expandable_segments:True`.

When the evaluation subprocess exits without parseable trace JSONL,
`src/sol_execbench/cli/main.py` writes a bounded diagnostic-only no-trace
sidecar and prints its path. This sidecar is for debugging stdout/stderr
framing failures and is deliberately separate from canonical trace JSONL.

## Test Markers

Core marker names are configured across `pyproject.toml` and
`tests/conftest.py`; additional environment-specific markers are registered in
`tests/conftest.py`.

| Marker | Purpose |
| --- | --- |
| `cpp` | Tests that compile HIP/C++ extensions. |
| `timing_serial` | GPU timing tests skipped by default unless selected with `-m timing_serial`. |
| `requires_rocm` | Tests that require a ROCm GPU visible through PyTorch. |
| `requires_rocm_dev` | Tests that require ROCm native extension headers under `/opt/rocm`. |
| `requires_ck` | Tests that require Composable Kernel headers. |
| `requires_rocwmma` | Tests that require rocWMMA headers. |
| `requires_rdna4` | Tests that require an AMD RDNA 4 GPU such as `gfx1200`. |
| `requires_cdna3` | Tests that require an AMD CDNA 3 GPU such as `gfx942`. |
| `requires_cutile` | Legacy NVIDIA cuTile marker skipped in this ROCm-only port. |

## Branch Conventions

The default branch is `main`. No feature branch naming pattern is documented in
the repository; keep branch names short and issue-oriented so each branch maps
to one approved GitHub issue.

Commit titles use:

```text
#<Issue Number> - <Commit Title>
```

Sign commits with DCO sign-off:

```bash
git commit -s -m "#123 - Fix trace parsing"
```

The repository pre-commit configuration includes a commit-message hook that
checks for a `Signed-off-by:` line.

## PR Process

Contributions should start from an approved GitHub issue.

- Link the approved issue in the pull request description.
- Keep the pull request focused on one concern.
- Include tests, lint, type checks, Docker checks, or GPU checks that were run.
- Update documentation when changing public commands, schemas, Docker targets,
  ROCm hardware support claims, scoring, timing, profiling, evidence semantics,
  or examples.
- Document hardware-specific assumptions in tests or PR notes.

## CI

The `.github/workflows/code-quality.yml` workflow runs on pushes and pull
requests for Python 3.12 and 3.13. It performs:

```bash
uv sync --locked --all-groups
uv run ruff check .
uv run ty check
uv run pytest tests/sol_execbench \
  --ignore=tests/sol_execbench/driver/test_eval_driver.py \
  --ignore=tests/sol_execbench/test_e2e.py
uv run pytest tests/examples/test_examples.py -k consistency
```

The workflow intentionally avoids tests that need live ROCm GPU execution.
