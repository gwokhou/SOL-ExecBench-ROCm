# SOL ExecBench ROCm Port

## What This Is

This project ports SOL ExecBench from its NVIDIA CUDA ecosystem implementation to
the AMD ROCm ecosystem. It is for researchers and developers who need to evaluate
LLM-generated GPU kernels on AMD GPUs using a benchmark standard that remains as
consistent as practical with the original SOL ExecBench paper and implementation.

## Core Value

Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware
while preserving the benchmark semantics and rigor of SOL ExecBench.

## Requirements

### Validated

- ✓ Existing CLI can load definitions, workloads, solutions, and emit trace results through `sol-execbench` — existing
- ✓ Existing driver stages problem files, compiles native solutions, and runs evaluation in a subprocess — existing
- ✓ Existing Pydantic schemas define public `definition.json`, `workload.jsonl`, `solution.json`, and trace contracts — existing
- ✓ Existing benchmark helpers cover input generation, correctness checks, timing, clock checks, environment capture, and reward-hack detection — existing
- ✓ Existing pytest suite covers schemas, benchmark helpers, driver subprocess behavior, examples, and Docker dependency checks — existing

### Active

- [ ] Replace NVIDIA/CUDA-specific runtime, Docker, compile, profile, analyze, and library paths with ROCm equivalents.
- [ ] Make ROCm the only supported runtime target; original NVIDIA GPU execution paths do not need to remain operational.
- [ ] Preserve benchmark semantics and standards from NVIDIA SOL ExecBench wherever ROCm hardware and software allow.
- [ ] Support ROCm-side equivalents for the original solution ecosystem, including HIP/C++, PyTorch on ROCm, Triton on ROCm, and replacements for CUTLASS, cuDNN, CuTe DSL, and cuTile where feasible.
- [ ] Run successfully on ROCm >= 7.0.
- [ ] Pass the migrated test suite on at least RDNA 4 and CDNA 3 hardware.
- [ ] Maintain license compliance with the repository license and all replacement ROCm dependencies.

### Out of Scope

- Maintaining CUDA/NVIDIA runtime compatibility — ROCm is the target platform for this port.
- Preserving NVIDIA-specific dependency implementations when a ROCm replacement is required — equivalent behavior matters more than retaining original code.
- Changing the benchmark's public schema without necessity — compatibility with existing SOL ExecBench problem formats should be preserved when possible.

## Context

The current codebase is NVIDIA-oriented. `docker/Dockerfile` uses CUDA 13.1.1,
cuDNN, CUTLASS, and CUDA environment variables. `pyproject.toml` pulls PyTorch
CUDA wheels and NVIDIA-specific packages such as `cuda-tile`,
`nvidia-cudnn-frontend`, and `nvidia-cutlass-dsl[cu13]`.

The architecture is a local CLI benchmark runner. `src/sol_execbench/cli/main.py`
loads problem JSON, `src/sol_execbench/driver/problem_packager.py` stages files
and injects GPU architecture flags, and `src/sol_execbench/driver/templates/`
contains the compile and eval subprocess scripts. Core schemas live in
`src/sol_execbench/core/data/`; correctness, IO, timing, clock locking, and
reward-hack defenses live in `src/sol_execbench/core/bench/`.

The high-risk areas are the eval driver, native compilation flow, GPU timing,
clock control, reward-hack defenses, hardware-specific test markers, and example
solution coverage. The port should reuse the existing CLI, schema, staging,
trace, and test structure wherever practical.

## Constraints

- **Platform**: ROCm >= 7.0 — the supported software baseline.
- **Hardware**: RDNA 4 and CDNA 3 — both architectures must pass the adapted test suite.
- **Compatibility**: Preserve SOL ExecBench benchmark semantics and public schemas unless a ROCm-specific change is unavoidable.
- **Scope**: NVIDIA/CUDA paths may be removed instead of maintained as a dual backend.
- **Licensing**: All retained and replacement code must comply with the repository LICENSE and third-party dependency obligations.
- **Quality**: Migrated tests, examples, Docker checks, and end-to-end evaluation must pass under ROCm.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| ROCm-only port | User explicitly chose to replace NVIDIA/CUDA paths rather than maintain dual backend compatibility. | - Pending |
| Broad solution ecosystem target | User chose to pursue ROCm equivalents for all original solution categories, not just HIP-only or a minimal subset. | - Pending |
| Adapted test suite is the completion gate | User defined done as passing the migrated existing tests on ROCm, rather than adding a separate parity suite as v1 scope. | - Pending |
| Preserve benchmark standards | The port should remain consistent with the NVIDIA SOL ExecBench paper and implementation standards wherever feasible. | - Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-21 after initialization*
