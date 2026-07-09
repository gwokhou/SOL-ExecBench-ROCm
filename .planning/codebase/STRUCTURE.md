---
generated_by: gsd-map-codebase
generated_on: 2026-07-09
last_mapped_commit: cc007cd3af3e5100f7d86f155a40d5e51ffb57e5
focus: arch
---

# Structure

## Top-Level Layout

- `src/sol_execbench/` - installable Python package.
- `tests/` - pytest suite, including package tests, examples tests, Docker
  dependency tests, sample fixtures, and marker-aware hardware skips.
- `docs/` - user docs, schema docs, ROCm notes, validation evidence, release
  notes, and internal validation records.
- `examples/` - runnable benchmark problem directories for PyTorch, Triton
  ROCm, HIP/C++, and selected ROCm libraries.
- `scripts/` - operator scripts for dataset migration, reports, validation,
  Docker, and batch execution.
- `scripts/internal/` - internal RDNA4 profiler, release, and report helpers.
- `docker/` - ROCm Dockerfile, entrypoint, and target matrix.
- `data/` - local downloaded benchmark assets; not intended for committed datasets.
- `.planning/` - GSD project state, roadmap, milestones, notes, and generated
  codebase maps.

## Package Layout

```text
src/sol_execbench/
  cli/
  core/
  data/
  driver/
  sol_score.py
```

`cli/` owns command-line UX. `core/` owns reusable implementation and contracts.
`data/` stores packaged static JSON such as AMD hardware model data. `driver/`
owns staging and generated subprocess runtime files.

## CLI Structure

```text
src/sol_execbench/cli/
  main.py
  commands/
  evaluation/
  sidecars/
```

- `main.py` defines the root Click command and dispatch wrapper.
- `commands/` contains metadata, environment, dataset, baseline, and root
  subcommand dispatch modules.
- `evaluation/` splits problem IO, compile phase, runtime phase, output
  handling, reporting, diagnostics, and sidecar writing.
- `sidecars/` contains CLI adapters for agent feedback, profile, static
  evidence, and common sidecar path behavior.

## Core Structure

```text
src/sol_execbench/core/
  bench/
  data/
  dataset/
  evidence/
  platform/
  reports/
  scoring/
```

- `bench/` is runtime evaluation logic: correctness, timing, input generation,
  reward-hack detection, profiler integration, static kernel evidence, profile
  summaries, and agent feedback.
- `data/` is schema and serialization: definitions, workloads, solutions,
  traces, dtypes, shapes, path access, and JSON utilities.
- `dataset/` is local dataset workflows: migration, inventory, readiness,
  execution closure, sharding, paper denominator, parity gaps, profiler timing
  coverage, run state, and runner helpers.
- `evidence/` is bounded runtime evidence references and collectors.
- `platform/` is ROCm environment, compatibility, Docker matrix, dependency
  matrix, and toolchain routing.
- `reports/` is report construction for consistency, matrix diffs, claim
  upgrades, trust summaries, evaluation stability, and report payloads.
- `scoring/` is AMD score/SOL calculations, bound estimates, graph annotations,
  official scoring, confidence, and solar derivation evidence.

## Driver Structure

```text
src/sol_execbench/driver/
  build_config.py
  eval_runtime_api.py
  eval_runtime_api_exports.py
  problem_packager.py
  staging.py
  trace_output.py
  templates/
```

- `problem_packager.py` stages normalized problem files and returns compile and
  evaluation commands.
- `staging.py` writes solution sources and safetensors assets.
- `build_config.py` probes local AMD gfx targets and injects HIP offload flags.
- `trace_output.py` parses driver stdout JSONL into trace models.
- `templates/build_ext.py` is copied into staging for native builds.
- `templates/eval_driver.py` is copied into staging for benchmark execution.

## Examples Layout

Each example problem directory contains `definition.json`, `workload.jsonl`,
one or more source files, a reference implementation when relevant, and a
`solution_*.json` file. Current families include:

- `examples/pytorch/` for PyTorch operator solutions.
- `examples/triton/` for Triton ROCm kernels.
- `examples/hip_cpp/` for native HIP/C++ extensions.
- `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and
  `examples/rocwmma/` for selected ROCm library categories.

Legacy example directories such as `examples/cudnn/`, `examples/cutlass/`,
`examples/cutile/`, and `examples/cute_dsl/` are present as migration or
negative-readiness context rather than supported public ROCm categories.

## Tests Layout

- `tests/conftest.py` registers markers and skips hardware-sensitive tests when
  ROCm devices, headers, Triton, safetensors, CK, or rocWMMA are unavailable.
- `tests/sol_execbench/cli/` mirrors `src/sol_execbench/cli/`.
- `tests/sol_execbench/core/` mirrors the core subpackages.
- `tests/examples/` validates example layout and CLI path behavior.
- `tests/docker/dependencies/` validates expected container dependencies and is
  skipped unless the `docker_dependency` marker is selected.
- `tests/samples/` contains solution and reward-hack fixtures.

There are also many legacy flat tests in `tests/sol_execbench/test_*.py`; newer
coverage increasingly mirrors the package subdirectories.

## Documentation Layout

Primary user docs are `README.md`, `docs/GETTING-STARTED.md`,
`docs/COOKBOOK.md`, `docs/CONFIGURATION.md`, `docs/DEVELOPMENT.md`,
`docs/TESTING.md`, and `docs/RESEARCHER-GUIDE.md`. Contract and schema docs
include `docs/EVALUATOR-CONTRACT.md`, `docs/definition.md`,
`docs/workload.md`, `docs/solution.md`, and `docs/trace.md`.

ROCm-specific docs include `docs/rocm.md`, `docs/rocm_timing.md`,
`docs/rocm_toolchain_routing.md`, and `docs/rocm_libraries.md`.
Internal validation records live under `docs/internal/`.

## Naming Conventions

Python modules use `snake_case`. Classes and Pydantic models use `PascalCase`.
Tests use `test_*` function names and often mirror the module under test.
Generated sidecar names use descriptive suffixes such as `.profile-summary.json`
or `.static-evidence.json` near trace outputs.
