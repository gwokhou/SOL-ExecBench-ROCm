---
generated_at: 2026-06-02
last_mapped_commit: 8019adc6295a78d4636037889245abcb3f9a52bb
focus: architecture
---

# Architecture

## System Shape

SOL ExecBench ROCm Port is a local benchmark package, not a hosted service. It evaluates a problem definition, workload rows, and a submitted solution by staging files, invoking an isolated Python evaluation driver, and emitting canonical Trace JSONL. Diagnostic sidecars can add environment, profiling, static-kernel, scoring, and release evidence, but Trace JSONL remains the primary benchmark artifact.

## Main Layers

- CLI layer: `src/sol_execbench/cli/main.py` parses inputs, loads schemas, packages problems, launches the subprocess driver, writes trace output, and attaches optional diagnostics.
- Data model layer: `src/sol_execbench/core/data/` defines Definition, Workload, Solution, Trace, dtype, shape, and public contract models.
- Benchmark runtime layer: `src/sol_execbench/core/bench/` contains correctness, input/output allocation, timing, clock policy, reward-hack checks, profiling helpers, static evidence helpers, and reusable driver runtime helpers.
- Driver layer: `src/sol_execbench/driver/` stages problems and uses templates under `src/sol_execbench/driver/templates/` for the actual evaluation subprocess.
- Dataset layer: `src/sol_execbench/core/dataset/` and `scripts/run_dataset.py` handle multi-problem discovery, execution closure, ready subsets, sharding, inventory, denominator accounting, and sidecar references.
- Scoring/evidence layer: `src/sol_execbench/core/scoring/`, `src/sol_execbench/core/consistency.py`, `src/sol_execbench/core/evaluation_stability.py`, `src/sol_execbench/core/trust_summary.py`, and related scripts build AMD-native diagnostic reports.
- Release layer: `scripts/build_prerelease_artifact_bundle.py`, `scripts/check_prerelease_readiness.py`, and `scripts/release_candidate_validation.py` enforce public prerelease and research preview boundaries.

## Data Flow

1. User runs `sol-execbench` with a problem directory or explicit `--definition`, `--workload`, and `--solution` paths.
2. CLI loads Pydantic models from JSON/JSONL and resolves source file contents relative to the solution JSON.
3. `ProblemPackager` stages definition, workload, solution, config, source files, and generated templates.
4. CLI runs the generated `eval_driver.py` in a subprocess.
5. The driver redirects non-JSON stdout to stderr, imports PyTorch, loads reference and user functions, checks reward-hack conditions, times workloads, checks correctness, and emits strict JSONL Trace records.
6. CLI parses Trace JSONL and writes user-requested output or diagnostic no-trace sidecars.
7. Optional scripts consume canonical traces and sidecars to produce Matrix, closure, scoring, consistency, stability, claim-upgrade, trust-summary, and prerelease artifacts.

## Execution Boundary

- The evaluation driver is a subprocess boundary, not a hardened sandbox.
- Static source review happens before importing user code in `src/sol_execbench/driver/templates/eval_driver.py`.
- Runtime reward-hack checks cover monkey patching, lazy outputs, thread injection, hidden stream patterns, and integrity changes through `src/sol_execbench/core/bench/reward_hack.py`.
- Native compile flags are restricted in `src/sol_execbench/core/data/solution.py` to avoid host path injection, response files, and dynamic linker control.
- Security documentation explicitly says untrusted submissions require container, VM, or isolated host controls.

## ROCm Boundary

- Public solution schema supports ROCm categories and rejects legacy CUDA/NVIDIA categories.
- The PyTorch device spelling may still use `cuda` because PyTorch ROCm exposes HIP devices through CUDA-compatible APIs.
- Docker, docs, and readiness gates distinguish RDNA 4 recorded evidence, deferred MI300X/CDNA3 full-suite validation, and unavailable CDNA4 validation.
- MI300X is the concrete CDNA3 `gfx942` hardware target, not a separate architecture peer.

## Extension Points

- Add new solution categories by updating `SupportedLanguages`, driver loading/build logic, examples, docs, and tests.
- Add new evidence sidecars by keeping Trace JSONL canonical and documenting authority class boundaries.
- Add new ROCm Docker targets through `docker/rocm-targets.json` and dependency preflight tests.
- Add new AMD scoring or bound families under `src/sol_execbench/core/scoring/` with explicit support and confidence semantics.
