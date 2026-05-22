# Codebase Concerns

**Analysis Date:** 2026-05-22

## Tech Debt

**ROCm Port Compatibility Namespaces:**
- Issue: ROCm execution still relies on PyTorch compatibility names such as `torch.cuda`, `cuda_events`, `extra_cuda_cflags`, and `at::cuda` because PyTorch ROCm exposes HIP devices through CUDA-named APIs.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/test_rocm_migration_residue_audit.py`
- Impact: New contributors can accidentally reintroduce real NVIDIA/CUDA dependencies while following compatibility names that are intentionally retained.
- Fix approach: Treat `tests/sol_execbench/test_rocm_migration_residue_audit.py` as the required guardrail whenever editing files that mention CUDA/NVIDIA terms. Add new residue classifications only when the compatibility use is ROCm-specific and documented.

**Regex-Based Reward-Hack Review:**
- Issue: Static source review blocks exploit families with regular expressions rather than syntax-aware analysis, so it can reject benign code that contains blocked tokens and miss equivalent behavior hidden through aliases or indirection.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Impact: Benchmark submission acceptance is fragile. False positives reduce usability, while false negatives can compromise timing and correctness measurements.
- Fix approach: Replace token regexes in `src/sol_execbench/core/bench/reward_hack.py` with Python AST checks for Python sources and explicit policy checks for native sources. Keep runtime integrity checks in `src/sol_execbench/driver/templates/eval_driver.py` as a second layer.

**Dataset Runner Source Mutation:**
- Issue: `scripts/run_dataset.py` mutates wrapped Python source text with `code.replace("stream", "strm")` and `reference_code.replace("stream", "strm")` to bypass the stream detector.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/bench/reward_hack.py`
- Impact: Reference or custom source semantics can change unexpectedly when identifiers or string literals contain `stream`, and the runner no longer evaluates the exact source supplied by the definition or user file.
- Fix approach: Remove source text rewriting from `scripts/run_dataset.py`. Narrow stream blocking in `src/sol_execbench/core/bench/reward_hack.py` to AST calls such as `torch.cuda.Stream`, `torch.cuda.stream`, and native stream API symbols.

**Generated Evaluation Driver Size:**
- Issue: The generated runtime driver combines problem loading, user-code import, static review, correctness loops, reward-hack checks, timing, trace emission, safetensors resolution, and clock-lock handling in one large script.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/test_e2e.py`
- Impact: Changes to one evaluation responsibility can alter another because the driver relies on module-level state and ordering. It is hard to isolate failures in subprocess logs.
- Fix approach: Extract pure helpers into importable modules under `src/sol_execbench/core/bench/` and keep `src/sol_execbench/driver/templates/eval_driver.py` as a thin orchestration script. Preserve subprocess-level tests in `tests/sol_execbench/driver/test_eval_driver.py`.

**Provisional AMD SOL Analyzer:**
- Issue: AMD speed-of-light bounds are derived from a conservative AST visitor and built-in hardware models with provisional or unvalidated status.
- Files: `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_score.py`, `tests/sol_execbench/test_amd_sol_bounds.py`, `docs/analysis.md`
- Impact: Derived score reports can be useful for ranking evidence but are not complete hardware-validation claims. Unsupported and inexact operations can produce scores with limited interpretability.
- Fix approach: Keep warnings in `src/sol_execbench/core/scoring/amd_score.py` mandatory. Extend `_CALL_ANALYZERS` in `src/sol_execbench/core/scoring/amd_sol.py` only with tests that assert coverage summary changes in `tests/sol_execbench/test_amd_sol_bounds.py`.

**Compatibility Example Categories:**
- Issue: Former CUDA library categories remain as compatibility examples using portable PyTorch implementations for categories such as `cute_dsl` and `cutile`.
- Files: `examples/cute_dsl/jamba_attn_proj/solution_cute_dsl.json`, `examples/cutile/jamba_attn_proj/solution_cutile.json`, `tests/examples/test_examples.py`, `tests/sol_execbench/test_rocm_library_examples.py`
- Impact: Example coverage preserves public surface area but does not prove optimized ROCm replacements for those categories.
- Fix approach: Treat these examples as compatibility fixtures. Add real ROCm-native examples under existing ROCm categories such as `examples/hip_cpp/`, `examples/hipblas/`, or future CK/rocWMMA directories when optimized implementations are added.

## Known Bugs

**Profiler Collection Can Hang Indefinitely:**
- Symptoms: Live `rocprofv3` evidence collection uses `subprocess.run()` without a timeout in the default runner.
- Files: `src/sol_execbench/core/bench/rocm_profiler.py`, `scripts/run_dataset.py`, `tests/sol_execbench/test_rocm_profiler.py`
- Trigger: `scripts/run_dataset.py --timing-evidence-dir ...` runs a profiler command that hangs because the target command, profiler, or GPU runtime stalls.
- Workaround: Use the injectable `runner` in tests and wrappers, or run dataset jobs with external process supervision.
- Fix approach: Add timeout support to `Rocprofv3CollectionRequest` and pass the dataset runner timeout through `collect_source_timing_evidence()`.

**First CSV Candidate Wins:**
- Symptoms: `_find_rocprofv3_csv()` sorts matching CSV files and returns the first path.
- Files: `src/sol_execbench/core/bench/rocm_profiler.py`, `tests/sol_execbench/test_rocm_profiler.py`
- Trigger: A profiler output directory contains stale files matching the same `output_file*.csv` prefix.
- Workaround: Use a clean timing evidence output directory for each run.
- Fix approach: Select the newest matching CSV, remove prior matching files before collection, or require an exact profiler output path in `Rocprofv3CollectionResult`.

**CLI Parses Partial Traces After Evaluation Process Failure:**
- Symptoms: The CLI exits based on parsed trace status when the evaluation subprocess returns non-zero but emits at least one trace.
- Files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/eval_driver.py`
- Trigger: `eval_driver.py` emits some workload traces and then crashes before all workloads finish.
- Workaround: Compare parsed trace count against expected workload count in callers that require full-run completion.
- Fix approach: In `src/sol_execbench/cli/main.py`, treat non-zero evaluation return codes as failure unless the parsed trace count equals `len(workloads)` and every workload has an explicit terminal evaluation.

## Security Considerations

**Untrusted Solution Execution:**
- Risk: Python solution files and compiled HIP/C++ extensions execute in the evaluator process with access to the staging directory, environment variables, imports, GPU, and local filesystem APIs not fully constrained by Python-level checks.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/core/bench/reward_hack.py`
- Current mitigation: Source paths reject absolute paths and `..` in `src/sol_execbench/core/data/solution.py`; static review blocks common file/process/network/dynamic-loader patterns in `src/sol_execbench/core/bench/reward_hack.py`; Python `load()` and `load_inline()` are replaced in `src/sol_execbench/driver/templates/eval_driver.py`.
- Recommendations: Run evaluation inside a hardened container or sandbox with minimal secrets, read-only benchmark inputs, restricted network, controlled environment variables, and resource limits. Do not rely on `src/sol_execbench/core/bench/reward_hack.py` as a complete sandbox.

**Compiler And Linker Flags Are User-Controlled:**
- Risk: Native solution `cflags`, `hip_cflags`, and `ld_flags` are passed directly to `torch.utils.cpp_extension.load()`.
- Files: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/driver/templates/build_ext.py`, `tests/sol_execbench/driver/test_build_ext.py`
- Current mitigation: Compile options are structured as lists, source paths are constrained, and compilation runs in a staging directory.
- Recommendations: Add an allowlist or denylist for dangerous flags, especially linker flags that load arbitrary libraries or alter runtime search paths. Keep native builds in isolated containers.

**Safetensors Path Resolution Can Read Outside Staging:**
- Risk: Workload safetensors paths are resolved and then loaded from absolute paths if provided or if no blob root match is found.
- Files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_io.py`
- Current mitigation: Shape and dtype are validated after loading, and staging roots are prioritized.
- Recommendations: For untrusted workloads, reject absolute safetensors paths or restrict them to approved blob roots before calling `safetensors.torch.load_file()`.

**Subprocess Logs May Expose Local Paths And Environment-Derived Details:**
- Risk: Failed CLI and build logs are saved or printed with stdout/stderr content from compilers, profilers, and user code.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/templates/build_ext.py`
- Current mitigation: Logs are local files, and `.env` files are not part of the normal load path.
- Recommendations: Scrub logs before publishing benchmark artifacts. Avoid running with secrets in the environment because user code executes in subprocesses inheriting `os.environ`.

## Performance Bottlenecks

**Correctness Runs Ten Full Rounds Per Workload:**
- Problem: Every workload performs ten correctness rounds with fresh input generation before timing.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/test_e2e.py`
- Cause: The driver intentionally catches nondeterministic and input-dependent errors.
- Improvement path: Make correctness rounds configurable in `BenchmarkConfig` while preserving a strict default for official benchmark runs.

**Cache Clearing Adds Extra GPU Work Around Every Timing Iteration:**
- Problem: Timing allocates a large cache buffer and zeroes it before warmup and measured calls.
- Files: `src/sol_execbench/core/bench/timing.py`, `tests/sol_execbench/core/bench/test_timing.py`
- Cause: `_get_empty_cache_for_benchmark()` sizes the buffer from reported L2 cache and `_clear_cache()` runs before each invocation.
- Improvement path: Keep cache-clearing semantics for official scoring, but add explicit labels and configuration for exploratory runs that do not need cold-cache timing.

**Input Pooling Can Consume Large VRAM For Many Tensors:**
- Problem: `ShiftingMemoryPoolAllocator` preallocates per-input and per-output pools sized by physical storage span plus iteration offsets.
- Files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/timing.py`, `tests/sol_execbench/core/bench/test_io.py`
- Cause: The allocator guarantees unique data pointers across warmup and timing iterations.
- Improvement path: Add memory budget checks before allocation and surface a clear `RUNTIME_ERROR` trace when the pool cannot fit.

**Dataset Runner Reinvokes CLI Per Problem:**
- Problem: `scripts/run_dataset.py` shells out to `sol-execbench` for each problem, and optional profiler evidence can shell out again through `rocprofv3`.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/rocm_profiler.py`
- Cause: Subprocess isolation keeps failures local but adds process startup, model validation, staging, and import overhead.
- Improvement path: Keep subprocess isolation for official runs. For development sweeps, add a controlled in-process mode only if it preserves trace semantics and reward-hack boundaries.

## Fragile Areas

**Evaluation Integrity Ordering:**
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Why fragile: Static review, reference import, critical-function snapshotting, user import, runtime integrity checks, correctness checks, and timing checks rely on a specific order.
- Safe modification: Add tests in `tests/sol_execbench/driver/test_eval_driver.py` for any new import, global, or reward-hack check. Preserve checks before user timing.
- Test coverage: Reward-hack behavior has focused subprocess tests, but coverage depends on known exploit fixtures and cannot prove all bypasses are blocked.

**Timing Semantics And Score Claims:**
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/timing_policy.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/scoring/amd_score.py`
- Why fragile: Device-event fallback, profiler-backed kernel activity, PyTorch operator attribution, and derived score claims carry different meanings.
- Safe modification: Keep policy metadata in every timing evidence payload and preserve warnings in score reports when profiler or hardware validation evidence is incomplete.
- Test coverage: Policy and parser tests exist in `tests/sol_execbench/test_timing_policy.py` and `tests/sol_execbench/test_rocm_profiler.py`; live profiler behavior depends on ROCm hardware and tooling.

**Schema Migration Guardrails:**
- Files: `src/sol_execbench/core/data/solution.py`, `tests/sol_execbench/core/data/test_solution.py`, `tests/sol_execbench/test_rocm_schema_build_audit.py`, `tests/sol_execbench/test_rocm_migration_residue_audit.py`
- Why fragile: Public schemas must reject legacy CUDA/NVIDIA values while keeping PyTorch ROCm compatibility names.
- Safe modification: Update schema tests and migration residue classification together. Do not add legacy language values to `SupportedLanguages`.
- Test coverage: Guardrail tests cover known legacy names and compile option keys, not every possible third-party schema variant.

**Hardware-Sensitive Test Selection:**
- Files: `tests/conftest.py`, `pyproject.toml`, `tests/docker/dependencies/test_pytorch_rocm.py`, `tests/docker/dependencies/test_rocm_runtime.py`
- Why fragile: Tests marked `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and `timing_serial` are skipped unless the right GPU and marker selection are present.
- Safe modification: Use markers consistently for GPU-dependent tests and record hardware-specific checks in docs or PR notes.
- Test coverage: Default `uv run pytest tests/` can pass while timing and architecture-specific behavior remains unexecuted on machines without the target GPU.

## Scaling Limits

**Single-GPU Evaluation Model:**
- Current capacity: Evaluation selects `cuda:0` when PyTorch reports a GPU and uses one subprocess per CLI invocation.
- Limit: Multi-GPU sharding and per-device scheduling are not modeled by the CLI or dataset runner.
- Scaling path: Add explicit device selection to `BenchmarkConfig` and pass it through `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/templates/eval_driver.py`, and `scripts/run_dataset.py`.

**Baseline Lookup Is Linear:**
- Current capacity: `ScoringBaselineArtifact.lookup()` scans all entries for each workload.
- Limit: Large release baseline artifacts cause repeated linear scans during suite scoring.
- Scaling path: Build an internal dictionary index in `src/sol_execbench/core/scoring/baseline_artifact.py` or in the caller that scores full suites.

**Dataset Output Accumulates Per-Problem Artifacts:**
- Current capacity: `scripts/run_dataset.py` writes traces, logs, optional timing evidence, optional score reports, and staged debug output under the selected output directory.
- Limit: Full-suite repeated runs can accumulate large artifact trees, especially with `--keep-staging` and profiler CSV output.
- Scaling path: Add retention controls and separate official artifact directories from temporary debugging directories.

## Dependencies at Risk

**PyTorch ROCm API Compatibility:**
- Risk: The project depends on PyTorch ROCm continuing to expose HIP devices through CUDA-named APIs and extension keywords.
- Impact: Timing, device selection, current-stream access in native samples, and HIP extension builds break if PyTorch changes compatibility APIs.
- Migration plan: Centralize PyTorch compatibility wrappers in `src/sol_execbench/core/bench/timing.py` and native build helpers; keep residue audit classifications explicit.

**rocprofv3 CSV Format:**
- Risk: `parse_rocprofv3_csv()` accepts a small set of normalized header names for name, domain, and duration fields.
- Impact: ROCm profiler version changes can silently parse fewer rows, producing zero kernel duration evidence.
- Migration plan: Add fixture CSVs for supported ROCm versions in `tests/sol_execbench/test_rocm_profiler.py` and fail collection when profiler-backed evidence contains no rows for a kernel-activity policy.

**ROCm Hardware Model Inputs:**
- Risk: Built-in `gfx1200` and `gfx942` models are provisional or unvalidated.
- Impact: Derived AMD-native score reports can be overinterpreted if warnings are ignored.
- Migration plan: Replace provisional entries in `src/sol_execbench/core/scoring/amd_sol.py` with release-validated hardware artifacts and add provenance tests.

## Missing Critical Features

**Hardened Evaluation Sandbox:**
- Problem: The evaluator detects known reward hacks but does not provide complete process, filesystem, environment, or network isolation.
- Blocks: Safe evaluation of untrusted submissions outside a controlled container or benchmark server.

**Release-Defined AMD Baselines:**
- Problem: `scripts/run_dataset.py --amd-score-report` falls back to reference latency when no scoring baseline artifact is supplied.
- Blocks: Release-quality AMD-native score claims without curated optimized baseline artifacts.

**Configurable Correctness Strictness:**
- Problem: Correctness rounds and timing cache-clearing behavior are fixed in code rather than represented as explicit benchmark policy fields.
- Blocks: Clear separation between official strict runs and fast development/debug runs.

## Test Coverage Gaps

**Live Profiler End-To-End Behavior:**
- What's not tested: Real `rocprofv3` collection against native HIP and Triton workloads across supported ROCm versions.
- Files: `src/sol_execbench/core/bench/rocm_profiler.py`, `scripts/run_dataset.py`, `tests/sol_execbench/test_rocm_profiler.py`
- Risk: Unit tests with injected runners pass while live profiler output, CSV naming, timeout behavior, or kernel-row parsing fails.
- Priority: High

**Sandbox Escape Resistance:**
- What's not tested: Comprehensive adversarial attempts using aliases, import indirection, native linker flags, environment reads, and filesystem side effects.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/templates/build_ext.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: Malicious or accidental submissions can alter measurement, leak local data, or consume unbounded resources.
- Priority: High

**Architecture-Specific ROCm Validation:**
- What's not tested: Full default test suite, examples, Docker dependency tests, and dataset slices on both RDNA 4 and CDNA 3 in ordinary local runs.
- Files: `tests/conftest.py`, `tests/docker/dependencies/`, `docs/internal/cdna3_validation_readiness.md`, `docs/internal/mi300x_validation_readiness.md`
- Risk: Default test results can miss hardware-specific failures and performance regressions.
- Priority: High

**Native Build Flag Safety:**
- What's not tested: Rejection or normalization of dangerous compiler/linker flags.
- Files: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/driver/templates/build_ext.py`, `tests/sol_execbench/driver/test_build_ext.py`
- Risk: Native builds can link unexpected libraries or alter runtime behavior.
- Priority: Medium

**Partial Trace Failure Handling:**
- What's not tested: Evaluation subprocess emits some traces and then exits non-zero.
- Files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/problem_packager.py`, `tests/sol_execbench/test_e2e.py`
- Risk: Callers can treat incomplete benchmark runs as complete result sets.
- Priority: Medium

---

*Concerns audit: 2026-05-22*
