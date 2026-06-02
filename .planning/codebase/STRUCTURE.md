---
generated_at: 2026-06-02
last_mapped_commit: 8019adc6295a78d4636037889245abcb3f9a52bb
focus: architecture
---

# Structure

## Root Layout

- `src/sol_execbench/` - package source.
- `tests/` - pytest suite, fixtures, samples, Docker dependency checks, and examples tests.
- `docs/` - public and internal documentation for schemas, ROCm behavior, release evidence, and claim boundaries.
- `examples/` - runnable problem examples grouped by backend/category.
- `scripts/` - dataset, evidence, release, Docker wrapper, and reporting scripts.
- `docker/` - ROCm Dockerfile, entrypoint, and target manifest.
- `data/` - placeholder for downloaded benchmark assets.
- `.planning/` - GSD project state, milestone archives, research notes, and codebase maps.

## Package Layout

- `src/sol_execbench/cli/`
  - `main.py` implements the benchmark CLI.
  - `baseline.py` implements baseline comparison CLI.
- `src/sol_execbench/core/`
  - `data/` contains schema models.
  - `bench/` contains evaluation runtime helpers.
  - `dataset/` contains dataset-scale execution and accounting helpers.
  - `scoring/` contains AMD-native scoring and SOL/SOLAR-derived diagnostic helpers.
  - top-level modules such as `compatibility.py`, `diagnostics.py`, `toolchain.py`, `runtime_evidence.py`, and `trust_summary.py` provide evidence and reporting surfaces.
- `src/sol_execbench/driver/`
  - `problem_packager.py` stages benchmark runs.
  - `templates/eval_driver.py` is copied into staging directories for subprocess execution.
  - `templates/build_ext.py` supports native extension builds.
- `src/sol_execbench/data/`
  - static package data, currently AMD hardware models.

## Test Layout

- `tests/sol_execbench/` holds most package tests.
- `tests/sol_execbench/core/bench/` covers timing, correctness, IO, clock, reward-hack, and runtime helpers.
- `tests/sol_execbench/core/data/` covers schema models.
- `tests/sol_execbench/driver/` covers packaging, build templates, and generated driver behavior.
- `tests/sol_execbench/samples/` and `tests/samples/` hold small fixture problems and adversarial samples.
- `tests/docker/dependencies/` checks ROCm container dependency assumptions.
- `tests/examples/` validates public examples and CLI path consistency.

## Documentation Layout

- User-facing docs: `README.md`, `docs/GETTING-STARTED.md`, `docs/CONFIGURATION.md`, `docs/TESTING.md`, `docs/DEVELOPMENT.md`, `docs/RESEARCHER-GUIDE.md`, and `docs/COOKBOOK.md`.
- Schema docs: `docs/definition.md`, `docs/workload.md`, `docs/solution.md`, and `docs/trace.md`.
- Evidence docs: `docs/rocm_timing.md`, `docs/static_kernel_evidence.md`, `docs/rocm_toolchain_routing.md`, `docs/analysis.md`, and `docs/v1_19_evidence_guide.md`.
- Release docs: `docs/prerelease_readiness.md`, `docs/prerelease_artifact_bundle.md`, `docs/public_prerelease.md`, `docs/research_preview.md`, and `docs/releases/`.
- Claim/provenance docs: `docs/CLAIMS.md`, `docs/compliance.md`, `docs/original_parity.md`, and `docs/provenance.md`.
- Internal validation docs: `docs/internal/`.

## Examples Layout

- ROCm-native examples: `examples/hip_cpp/`, `examples/hipblas/`, `examples/miopen/`, `examples/ck/`, and `examples/rocwmma/`.
- Python/Triton examples: `examples/pytorch/` and `examples/triton/`.
- Legacy NVIDIA example directories such as `examples/cutlass/`, `examples/cudnn/`, `examples/cutile/`, and `examples/cute_dsl/` remain as compatibility/residue references and are not supported ROCm categories unless explicitly migrated.

## Naming Patterns

- Problem directories contain `definition.json`, `workload.jsonl`, optional `config.json`, and one or more `solution*.json` files.
- Source modules use `snake_case`.
- Test files use `test_*.py`, with descriptive names matching the behavior under test.
- Evidence/reporting scripts generally use `report_*.py` or explicit action names under `scripts/`.
