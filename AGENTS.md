# Repository Guidelines

## Project Structure & Module Organization

SOL ExecBench is a Python package under `src/sol_execbench/`. The CLI entry point is
`sol_execbench.cli:cli`, exposed as `sol-execbench` by `pyproject.toml`. Core
benchmarking, data models, driver code, and score utilities live below this package.

Tests are in `tests/`, with package tests under `tests/sol_execbench/` and example
coverage in `tests/examples/`. Documentation is in `docs/`, runnable examples are in
`examples/`, helper scripts are in `scripts/`, Docker support is in `docker/`, and
downloaded benchmark assets belong in `data/`.

## Build, Test, and Development Commands

- `uv sync --all-groups`: install runtime and development dependencies.
- `uv run sol-execbench <problem_dir> --solution <solution-path>`: run the benchmark CLI.
- `uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark --limit 5`: run a small
  dataset batch.
- `uv run pytest tests/`: run the full test suite.
- `uv run pytest tests/sol_execbench/test_e2e.py`: run one test file.
- `uv run --with ruff ruff check .`: lint the repository when Ruff is not already installed.
- `uv run --with ruff ruff format .`: format Python files when Ruff is not already installed.
- `./scripts/run_docker.sh --build`: build and enter the Docker environment used for
  GPU evaluation.

## Coding Style & Naming Conventions

Use Python 3.12+ and follow the style enforced by Ruff. Keep changes consistent with
nearby modules and avoid broad refactors in focused fixes. Use `snake_case` for
functions, variables, and modules; `PascalCase` for classes and Pydantic models; and
clear, descriptive test names such as `test_rejects_invalid_solution_schema`.

Generated data, downloaded datasets, and examples are excluded from Ruff checks. Do
not commit local cache, build, or downloaded benchmark output.

## Testing Guidelines

Pytest is the test framework. Place new tests next to related coverage under
`tests/sol_execbench/`, or under `tests/examples/` for example workflows. Use existing
markers for environment-sensitive tests, including `cpp` for compiled extension tests,
`requires_rocm` for ROCm GPU behavior, and architecture-specific markers such as
`requires_rdna4` or `requires_cdna3`. `requires_cutile` is a legacy NVIDIA cuTile
marker that is skipped in this ROCm-only port. Prefer small unit tests
for schema and driver logic, and add integration coverage when changing subprocess
evaluation or GPU execution behavior.

## Commit & Pull Request Guidelines

Contributions should start from an approved GitHub issue. Commit titles use imperative
mood and the project format:

```text
#<Issue Number> - <Commit Title>
```

Sign commits with DCO sign-off, for example `git commit -s -m "#123 - Fix trace parsing"`.
Keep PRs focused on one concern, include a clear description of behavior changes, link
the issue, and list the tests or GPU checks you ran. New components should include
documentation and tests.

## Security & Configuration Tips

Do not commit proprietary kernels, credentials, Hugging Face tokens, or downloaded
datasets. GPU evaluation may require Docker, ROCm-capable AMD hardware, ROCm
drivers, and access to `/dev/kfd` and `/dev/dri`; document any hardware-specific
assumptions in tests or PR notes.

<!-- GSD:project-start source:.planning/PROJECT.md -->
## Project

**SOL ExecBench ROCm Port**

This project ports SOL ExecBench from its NVIDIA CUDA ecosystem implementation to
the AMD ROCm ecosystem. It is for researchers and developers who need to evaluate
LLM-generated GPU kernels on AMD GPUs using a benchmark standard that remains as
consistent as practical with the original SOL ExecBench paper and implementation.

**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware
while preserving the benchmark semantics and rigor of SOL ExecBench.

### Constraints

- **Platform**: ROCm >= 7.0 — the supported software baseline.
- **Hardware**: RDNA 4 and CDNA 3 — both architectures are project targets for the adapted test suite; RDNA 4 has recorded validation, while full CDNA 3 validation remains deferred.
- **Compatibility**: Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable.
- **Scope**: NVIDIA/CUDA paths may be removed instead of maintained as a dual backend.
- **Licensing**: All retained and replacement code must comply with the repository LICENSE and third-party dependency obligations.
- **Quality**: Migrated tests, examples, Docker checks, and end-to-end evaluation must pass under ROCm.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:.planning/codebase/STACK.md -->
## Technology Stack

## Runtime
## Primary Technologies
- Python package source lives in `src/sol_execbench/`.
- CLI layer uses Click and Rich in `src/sol_execbench/cli/main.py`.
- Data schemas use Pydantic v2 models in `src/sol_execbench/core/data/`.
- GPU evaluation uses PyTorch ROCm, Triton ROCm, HIP/C++, hipBLAS, and ROCm
  library candidate categories.
- Native HIP/C++ builds go through `torch.utils.cpp_extension`.
- Timing uses HIP-backed PyTorch device events, with optional `rocprofv3`
  evidence collection.
## Dependency Configuration
## Packaging
## Container
## Local Commands
- `uv sync --all-groups` installs dependencies.
- `uv run sol-execbench <problem_dir> --solution <solution-path>` runs one problem.
- `uv run pytest tests/` runs tests.
- `uv run --with ruff ruff check .` lints when Ruff is not already installed.
- `uv run --with ruff ruff format .` formats when Ruff is not already installed.
- `./scripts/run_docker.sh --build` builds and enters the GPU container.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:.planning/codebase/CONVENTIONS.md -->
## Conventions

## Python Style
## Source Headers
## Type And Model Style
## Naming
- Modules, functions, and variables use `snake_case`.
- Classes and Pydantic models use `PascalCase`.
- Enum classes use `PascalCase`; enum members use uppercase names with string
- Tests use descriptive `test_*` names and class groupings like
## Error Handling
## Subprocess Boundaries
## Security Patterns
## Documentation
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:.planning/codebase/ARCHITECTURE.md -->
## Architecture

## System Shape
## Main Layers
## Data Flow
## Execution Isolation
## Extension Points
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
