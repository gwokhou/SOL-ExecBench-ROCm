# Codebase Concerns

**Analysis Date:** 2026-06-01

## Status Legend

- Fixed: concern was closed by completed milestone work.
- Narrowed: concern was reduced to a smaller remaining responsibility or moved
  behind a clearer helper boundary.
- Externally blocked/deferred: concern requires hardware, operating-system,
  infrastructure, paper-scale validation, or hosted-service evidence outside
  the current code-actionable scope.

## Tech Debt

**Dataset-scale orchestration remains concentrated in one script path:**
- Issue: `scripts/run_dataset.py` owns argument parsing, dataset discovery, ready-subset filtering, resume/reuse behavior, closure records, derived report fan-out, timing evidence, summary writing, and subprocess invocation. The script delegates helpers into `src/sol_execbench/core/dataset/runner.py` and `src/sol_execbench/core/dataset/run_closure.py`, but the main control flow still spans many concerns.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/dataset/run_closure.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`
- Impact: Small edits to workload selection, `problem_output_dir`, trace reuse, or sidecar generation can change multiple modes at once. Regression risk is highest for ready-subset execution and derived evidence completeness.
- Fix approach: Continue extracting pure helpers into `src/sol_execbench/core/dataset/` and keep `scripts/run_dataset.py` as a thin CLI wrapper. New dataset behavior should land first as helper functions with focused tests under `tests/sol_execbench/`, then be wired through the script.

**Large derived-evidence modules are hard to review safely:**
- Issue: AMD bound and SOLAR derivation logic is implemented in large files with many internal taxonomies and fallback paths.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`
- Impact: Changes can silently affect claim boundaries, unsupported/degraded/scored classification, or derived score eligibility across many workload families.
- Fix approach: Add new operator-family behavior through narrow functions and fixture-backed tests in `tests/sol_execbench/fixtures/solar_derivation/`, `tests/sol_execbench/test_solar_derivation_evidence.py`, and `tests/sol_execbench/test_amd_sol_v2.py`. Avoid broad edits that touch graph extraction, work estimates, and score aggregation in the same change.

**Compatibility naming leaks into ROCm-only paths:**
- Issue: ROCm execution necessarily uses PyTorch's `torch.cuda` compatibility namespace, but legacy CUDA/NVIDIA names also remain in sample filenames and former-library examples.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/runtime_evidence.py`, `tests/sol_execbench/samples/rmsnorm/solution_cuda.json`, `tests/sol_execbench/samples/flux_rope/solution_cuda.json`, `examples/cudnn/softmax/solution_cudnn.json`, `examples/cutlass/gemm/solution_cutlass.json`, `examples/cutile/jamba_attn_proj/solution_cutile.json`
- Impact: Users and future agents can confuse compatibility API names or compatibility examples with active NVIDIA runtime support.
- Fix approach: Preserve intentional `torch.cuda` compatibility usage, but keep user-facing docs and tests explicit. Use guardrail tests such as `tests/sol_execbench/test_rocm_migration_residue_audit.py`, `tests/sol_execbench/test_rocm_library_examples.py`, and `tests/sol_execbench/test_public_contract_guardrails.py` when changing naming, examples, or schema text.

**Generated local artifacts appear throughout the working tree:**
- Issue: `__pycache__` directories are present under `src/`, `tests/`, and `scripts/`. They are ignored by `.gitignore`, but they increase scan noise and can hide stale local state during manual inspection.
- Files: `.gitignore`, `src/sol_execbench/__pycache__/`, `src/sol_execbench/core/__pycache__/`, `tests/sol_execbench/__pycache__/`, `scripts/__pycache__/`
- Impact: Tooling output becomes harder to read, and codebase maps can accidentally include generated paths if globs are not filtered.
- Fix approach: Keep generated artifacts ignored and exclude `**/__pycache__/**` in repository scans. Do not base implementation decisions on cached bytecode files.

## Known Bugs

**Existing successful dataset traces can be reused without provenance outside closure mode:**
- Symptoms: When `traces.json` exists, no failures are detected, `--rerun` is absent, and no `execution_closure_path` is active, dataset runs reuse the prior pass with reason `existing_pass`.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/dataset/runner.py`
- Trigger: Run `scripts/run_dataset.py` once, then change workload selection, benchmark config, solution mode, timing options, or repository code and run again without `--rerun` and without a ready-subset execution closure.
- Workaround: Use `--rerun` for non-closure dataset runs when any input, config, code, or evidence requirement changed. Prefer ready-subset/closure mode when reuse provenance matters.

**Reference-baseline score fallback can look more authoritative than it is:**
- Symptoms: AMD-native score reports use trace reference latency as a provisional baseline when no scoring baseline artifact is provided.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/scoring/amd_score.py`
- Trigger: Generate `--amd-score-report` without `--scoring-baseline`.
- Workaround: Provide a release-defined scoring baseline artifact through `--scoring-baseline`; otherwise treat the report as provisional derived evidence and preserve `REFERENCE_BASELINE_WARNING`.

**Static source review can reject benign submitted code patterns:**
- Symptoms: Submitted sources are blocked when they contain file I/O, dynamic imports, process/network access, stream APIs, semantic caches, or precision downgrades matching conservative static rules.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`
- Trigger: A legitimate solution uses identifiers or implementation techniques that match a blocked reward-hack pattern.
- Workaround: Keep benchmark solution code simple and self-contained. When changing review rules, add positive and negative tests in `tests/sol_execbench/core/bench/test_reward_hack.py` so fraud defenses remain strict without blocking required ROCm patterns.

## Security Considerations

**Evaluation is not a hard sandbox for untrusted code:**
- Risk: User solution code is imported and executed in a subprocess with access to the staging directory, Python runtime, inherited environment, and visible ROCm device resources. Static review blocks many risky patterns, but it is not process isolation.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/cli/main.py`, `SECURITY.md`, `docs/RESEARCHER-GUIDE.md`
- Current mitigation: Evaluation runs in a child process, dynamic `torch.utils.cpp_extension.load()` is patched off for Python solutions, stdout is separated from trace JSONL, static source review blocks known exploit families, and reward-hack checks detect monkey-patching, lazy outputs, and thread injection.
- Recommendations: Continue documenting this as a benchmark harness, not a sandbox. Do not pass secrets into the evaluation environment. For untrusted submissions, run inside a disposable container or VM with minimal environment variables and only required ROCm device access.

**Diagnostic logs and sidecars can capture sensitive local context:**
- Risk: Failure logs and no-trace diagnostics record bounded stdout/stderr tails; environment/runtime sidecars can include local hardware, library, path, and command metadata.
- Files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/runtime_evidence.py`, `src/sol_execbench/core/environment.py`
- Current mitigation: CLI diagnostic tails are bounded, dataset CLI logs are capped by `CLI_LOG_LIMIT`, and documentation warns against credentials and downloaded datasets.
- Recommendations: Keep log tails bounded, avoid dumping full environment maps, redact token-like values before writing shareable artifacts, and avoid embedding absolute paths in new reports when a relative evidence reference is sufficient.

**Docker execution requires privileged device exposure:**
- Risk: ROCm evaluation requires `/dev/kfd`, `/dev/dri`, group access, and GPU runtime settings. Broadening the Docker wrapper can expose more of the host than the benchmark needs.
- Files: `scripts/run_docker.sh`, `docker/Dockerfile`, `src/sol_execbench/core/docker_matrix.py`, `tests/sol_execbench/test_run_docker_matrix_script.py`, `tests/sol_execbench/test_docker_matrix_preflight.py`
- Current mitigation: Docker target selection and preflight checks are tested, unknown targets are rejected unless explicitly overridden, and documentation describes ROCm device requirements.
- Recommendations: Keep host mounts and privileges minimal. Add tests before changing Docker flags, target manifest behavior, or runtime preflight classification.

## Performance Bottlenecks

**Per-workload correctness and timing are intentionally expensive:**
- Problem: Each workload runs 10 correctness rounds, then timing warmups and repetitions. GPU memory is garbage-collected and cache-cleared between workloads.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/correctness.py`
- Cause: The harness prioritizes nondeterminism detection, output integrity, and stable timing over speed.
- Improvement path: Keep defaults conservative. If adding fast modes, make them explicit config fields in `BenchmarkConfig`, surface them in trace/config sidecars, and test that fast-mode traces cannot be mistaken for canonical benchmark output.

**Dataset runner subprocess fan-out is serial per problem:**
- Problem: `scripts/run_dataset.py` invokes `sol-execbench` per problem and writes per-problem artifacts synchronously.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/runner.py`
- Cause: Serial execution keeps GPU use and artifact writes simple, but large benchmark slices take longer.
- Improvement path: Add sharding or queue-based parallelism only with explicit output isolation, GPU resource controls, and closure provenance fields. Existing sharding helpers and tests should be extended under `src/sol_execbench/core/dataset/sharding.py` and `tests/sol_execbench/test_dataset_sharding.py`.

**Optional profiling can double-run evaluation:**
- Problem: `--profile rocprofv3` attempts profiled execution first and falls back to normal evaluation when profiling is unavailable or fails.
- Files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/timing_policy.py`
- Cause: Profiling is diagnostic-only and may be unavailable on a host, so fallback preserves normal benchmark execution.
- Improvement path: Keep profile sidecars explicit about `skipped_reason` and `failed_reason`. Avoid treating fallback event timing as profiler-backed evidence in new reports.

## Fragile Areas

**Trace JSONL parsing tolerates non-JSON output and can hide noisy failures:**
- Files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/dataset/runner.py`
- Why fragile: `convert_stdout_to_traces()` parses only lines beginning with `{`; dataset `run_cli()` ignores JSON decode errors. This is necessary because libraries can emit noise, but it means malformed trace output can degrade into "no traces" diagnostics instead of a precise schema error.
- Safe modification: Preserve strict `Trace` model parsing for canonical records, keep bounded diagnostics for no-trace cases, and add tests for malformed stdout/stderr behavior in `tests/sol_execbench/driver/test_problem_packager.py` and `tests/sol_execbench/test_cli_environment_snapshot.py`.
- Test coverage: Existing tests cover no-trace diagnostics and problem packager behavior; add focused cases for any new trace-output mode.

**Claim boundary wording is heavily guarded and easy to regress:**
- Files: `docs/CLAIMS.md`, `docs/analysis.md`, `docs/rocm.md`, `.planning/PROJECT.md`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/dataset/paper_denominator.py`, `tests/sol_execbench/test_public_contract_guardrails.py`
- Why fragile: The ROCm port must preserve distinctions between local derived evidence, paper denominator accounting, hardware validation, leaderboard readiness, NVIDIA/B200 equivalence, and unsupported/deferred states.
- Safe modification: Update docs, reports, and guardrail tests together. Do not collapse `ready`, `blocked`, `unsupported`, `deferred`, `evidence_missing`, `attempted_passed`, and `skipped_existing_pass` into a generic "skipped" or "validated" bucket.
- Test coverage: Guardrail tests are broad; maintain them when touching scoring, claims, readiness, denominator reports, or release docs.

**Native HIP/C++ build path depends on local ROCm tool discovery:**
- Files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/core/data/solution.py`, `tests/sol_execbench/driver/test_build_ext.py`, `tests/sol_execbench/driver/test_problem_packager.py`
- Why fragile: `LOCAL` target handling probes `rocm_agent_enumerator` and `rocminfo`, injects offload architecture flags, then relies on `torch.utils.cpp_extension` for native builds.
- Safe modification: Keep target injection deterministic, preserve explicit `hip_cflags`, and test probe fallbacks without requiring live ROCm hardware.
- Test coverage: Unit tests cover schema and build template behavior; live compiler/GPU paths still depend on marked ROCm tests.

**Hardware-sensitive tests skip by environment marker:**
- Files: `tests/conftest.py`, `pyproject.toml`, `tests/examples/test_examples.py`, `tests/examples/test_rocm_cli_paths.py`, `tests/sol_execbench/core/bench/test_timing.py`
- Why fragile: `requires_rocm`, `requires_rocm_dev`, `requires_rdna4`, `requires_cdna3`, `requires_ck`, `requires_rocwmma`, and `timing_serial` can skip critical GPU behavior on CPU-only hosts.
- Safe modification: Pair hardware tests with CPU-safe static/unit coverage where possible, but do not remove live ROCm tests for subprocess, native compilation, or timing behavior.
- Test coverage: CPU-safe tests are extensive; full confidence for timing/native execution still requires ROCm hosts and architecture-specific runs.

## Scaling Limits

**Full paper-scale validation is not represented as complete:**
- Current capacity: The codebase supports bounded ready-subset runs and derived reports, with claim boundaries marking full 235-problem validation as false unless direct evidence exists.
- Limit: Paper-scale or leaderboard-style claims break if reports aggregate discovered/parsed/ready workloads as validated workloads.
- Scaling path: Use `src/sol_execbench/core/dataset/paper_denominator.py`, `src/sol_execbench/core/dataset/execution_closure.py`, and `scripts/run_dataset.py` closure outputs to keep denominator, readiness, execution, and evidence completeness separate.

**CDNA 3 validation remains deferred by policy:**
- Current capacity: Schema, docs, and tests recognize CDNA 3 targets such as `gfx940`, `gfx941`, and `gfx942`.
- Limit: Real CDNA 3 full-suite validation is not claimed without recorded full-suite evidence.
- Scaling path: Run the marked CDNA 3 suite on real `gfx94*` hardware, capture environment sidecars, clock/timing evidence, pass/skip/fail counts, and update docs/tests that currently enforce deferred wording.

**External dataset assets are required for some workloads:**
- Current capacity: Workloads can refer to safetensors paths and the evaluation driver searches staging plus `FLASHINFER_TRACE_DIR`.
- Limit: Dataset runs fail or are classified blocked when FlashInfer trace blobs or benchmark assets are not present.
- Scaling path: Keep downloaded assets under `data/`, use readiness reports to separate missing assets from unsupported runtime paths, and avoid committing raw datasets or local download output.

## Dependencies at Risk

**ROCm wheel and package index coordination is narrow:**
- Risk: `pyproject.toml` pins PyTorch ROCm wheels and `triton-rocm` through explicit indexes. These packages are platform- and index-sensitive.
- Impact: `uv sync --all-groups` can fail or resolve unusable wheels when the ROCm index, Python version, platform marker, or ROCm release changes.
- Migration plan: Update `pyproject.toml`, lockfile, Docker image defaults, and dependency matrix tests together. Run dependency preflight tests under `tests/docker/dependencies/` and compatibility matrix tests under `tests/sol_execbench/`.

**Toolchain probes depend on host-installed ROCm tools:**
- Risk: `rocm_agent_enumerator`, `rocminfo`, `rocprofv3`, `rocm-smi`, HIP headers, CK, and rocWMMA may be missing or version-skewed.
- Impact: Local target detection, profiling, clock locking, native compilation, and example coverage can skip or fail depending on host setup.
- Migration plan: Keep probes explicit and nonfatal where evidence is optional. Use reason-coded sidecars from `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/core/docker_matrix.py`.

## Missing Critical Features

**No hard isolation boundary for arbitrary submissions:**
- Problem: The harness evaluates user code for benchmarking but does not enforce OS-level sandboxing.
- Blocks: Safe execution of untrusted third-party submissions in shared developer environments.

**No authoritative release baseline by default for AMD-native score reports:**
- Problem: Without `--scoring-baseline`, score generation falls back to reference latency and warns that the baseline is provisional.
- Blocks: Release-quality score comparisons from being treated as stable unless a baseline artifact is supplied and referenced.

**CDNA 3 real-hardware closure is not complete:**
- Problem: CDNA 3 support remains code/schema/readiness-level until full adapted suite validation is recorded.
- Blocks: CDNA 3 hardware-validation claims and MI300X-level release claims.

## Test Coverage Gaps

**Live ROCm execution and timing behavior are environment-gated:**
- What's not tested: Full subprocess evaluation, native HIP/C++ compilation, timing stability, clock locking, and architecture-specific kernels on every ordinary CI run.
- Files: `tests/conftest.py`, `tests/examples/test_rocm_cli_paths.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: CPU-safe tests can pass while ROCm driver, compiler, profiler, or architecture-specific behavior regresses.
- Priority: High

**Dataset runner ordinary reuse mode lacks provenance enforcement tests:**
- What's not tested: Non-closure `existing_pass` reuse against changed workload/config/solution provenance.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/run_closure.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`
- Risk: Dataset summaries can mix stale traces with new denominator or scoring context when closure mode is not used.
- Priority: Medium

**Security boundary tests cover known reward hacks, not complete sandbox escapes:**
- What's not tested: Full OS-level containment of arbitrary code, environment exfiltration attempts, filesystem traversal beyond static patterns, and native extension side effects.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: Users may overestimate the harness as a sandbox; new Python or native patterns can bypass static review without violating current tests.
- Priority: High

**Former NVIDIA library compatibility examples are mostly policy-guarded:**
- What's not tested: Full runtime parity for former `cutlass`, `cudnn`, and `cutile` categories because they are compatibility examples or skipped legacy paths in the ROCm port.
- Files: `examples/cutlass/gemm/`, `examples/cudnn/softmax/`, `examples/cutile/jamba_attn_proj/`, `tests/sol_execbench/test_rocm_library_examples.py`, `tests/sol_execbench/test_rocm_library_readiness_docs.py`
- Risk: Users can misread compatibility examples as active library support.
- Priority: Medium

---

*Concerns audit: 2026-06-01*
