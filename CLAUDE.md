# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SOL ExecBench ROCm Port evaluates custom GPU kernel solutions on AMD ROCm
hardware. It supports PyTorch ROCm, Triton ROCm, HIP/C++, and selected ROCm
library categories while rejecting legacy CUDA/NVIDIA runtime metadata.

## Build & Run Commands

```bash
# Install dependencies (uses uv package manager)
uv sync --all-groups

# Run the CLI
uv run sol-execbench <problem_dir> --solution solution.json

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/sol_execbench/core/data/test_definition.py

# Run tests matching a keyword
uv run pytest tests/ -k "test_correctness"

# Run ROCm-sensitive tests when ROCm hardware is available
uv run pytest tests/sol_execbench/test_e2e.py

# Lint and format
uv run ruff check .
uv run ruff format .
```

## Architecture

The codebase has three layers:

1. **CLI** (`src/sol_execbench/cli/main.py`) — Click-based CLI entry point (`sol-execbench`). Parses args, invokes the driver, displays Rich tables with results.

2. **Driver** (`src/sol_execbench/driver/`) — Orchestrates evaluation. `ProblemPackager` stages problem files (definition, workloads, solution sources) into a temp directory, then runs compilation and evaluation as **subprocesses**. The eval driver template (`src/sol_execbench/driver/templates/eval_driver.py`) is the self-contained script that runs inside the subprocess; it imports torch, loads inputs, executes kernels, computes correctness/performance, and emits JSONL traces to stdout.

3. **Core** (`src/sol_execbench/core/`) — Package modules for data models, benchmarking, scoring, diagnostics, and reporting:
   - `src/sol_execbench/core/data/` — Pydantic v2 models for Definition, Solution, Workload, Trace, shapes, and dtypes.
   - `src/sol_execbench/core/bench/` — Benchmarking utilities: HIP-backed event timing, numerical correctness computation, ROCm clock locking, and reward-hack detection.

### Key execution flow

CLI → ProblemPackager stages files to temp dir → subprocess compiles HIP/C++ when needed → subprocess runs eval_driver.py → eval_driver outputs JSONL traces to stdout → CLI parses traces and displays results.

### Two calling conventions

Kernel solutions use either **destination-passing style** (DPS, modifies pre-allocated outputs in-place) or **return-value style** (returns outputs directly). The eval driver handles both.

## Test Markers & Fixtures

- `@pytest.mark.requires_rocm` — requires a ROCm GPU visible through PyTorch
- `@pytest.mark.cpp` — compiles HIP/C++ extensions (slow)
- `@pytest.mark.requires_rdna4` / `@pytest.mark.requires_cdna3` — require specific AMD architecture classes
- `@pytest.mark.requires_cutile` — legacy NVIDIA cuTile marker skipped in this ROCm-only port
- `tmp_cache_dir` fixture — isolated build cache per test via `SOLEXECBENCH_CACHE_PATH`

## Style

Ruff for linting and formatting. Rule E741 is ignored. `data/` and `examples/`
are excluded from lint.

## Commits

Format: `#<Issue> - <Title>` with DCO sign-off (`git commit -s`).
