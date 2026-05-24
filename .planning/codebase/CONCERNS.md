# Codebase Concerns

**Analysis Date:** 2026-05-24

## Tech Debt

**Generated evaluation driver is a large script template:**
- Issue: The benchmark runtime is concentrated in a single generated script template that handles stdout redirection, reference execution, user-module import, reward-hack checks, correctness loops, timing, and trace emission.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/cli/main.py`
- Impact: Changes to evaluation semantics require editing a copied script boundary, and regressions can appear only after staging and subprocess execution. Shared logic cannot be imported directly without preserving the generated-driver isolation model.
- Fix approach: Keep `eval_driver.py` behavior covered by subprocess-style tests in `tests/sol_execbench/driver/test_eval_driver.py`; extract only pure helpers into importable modules when the helper has no dependency on staging globals, fd redirection, or user-code import ordering.

**Dataset runner mixes orchestration, filtering, scoring, closure, and evidence generation:**
- Issue: `scripts/run_dataset.py` is over 1,700 lines and owns discovery, solution wrapping, CLI invocation, timing evidence, AMD score sidecars, ready-subset closure, and report writing.
- Files: `scripts/run_dataset.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`, `tests/sol_execbench/test_run_dataset_amd_score.py`
- Impact: Small changes to dataset execution can affect report formats, closure status accounting, and derived scoring output. It is hard to reason about partial runs and rerun behavior without reading distant helper functions.
- Fix approach: Preserve the current CLI surface, but move cohesive units into package modules under `src/sol_execbench/core/dataset/` or `src/sol_execbench/core/scoring/`; keep script-level code as argument parsing and orchestration.

**Scoring and SOLAR derivation modules are dense, heuristic-heavy files:**
- Issue: AMD bound graph extraction, per-operator estimates, and SOLAR derivation are implemented as large heuristic modules with many private helpers and fallback paths.
- Files: `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_sol.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`
- Impact: Operation-family additions can silently change score eligibility, warning semantics, or derived evidence structure across many helpers. The code is test-covered, but still fragile because correctness depends on consistent warning names and evidence fields.
- Fix approach: Add new operation support through focused fixtures in `tests/sol_execbench/test_amd_bound_graph.py`, `tests/sol_execbench/test_amd_bound_estimates.py`, and `tests/sol_execbench/test_solar_derivation_evidence.py`; avoid broad refactors that combine graph extraction, work estimation, and evidence serialization.

**Legacy naming remains in ROCm-compatible paths:**
- Issue: ROCm timing and examples intentionally use PyTorch's `torch.cuda` compatibility namespace, and some test/example artifacts retain CUDA-oriented names.
- Files: `src/sol_execbench/core/bench/timing.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/samples/rmsnorm/solution_cuda.json`, `tests/sol_execbench/samples/flux_rope/solution_cuda.json`, `docs/rocm_timing.md`
- Impact: New contributors can mistake compatibility names for NVIDIA runtime support or remove necessary ROCm-compatible `torch.cuda` calls. Residue audits reduce this risk but do not remove the cognitive load.
- Fix approach: Keep compatibility names documented in nearby comments and docs; when adding new ROCm code, prefer HIP/ROCm names unless the PyTorch API really is exposed only through `torch.cuda`.

**Reference-solution wrapping mutates source text to avoid static review:**
- Issue: The dataset runner replaces the literal string `stream` with `strm` when wrapping reference/custom Python solutions.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/bench/reward_hack.py`
- Impact: This can change semantics for legitimate code that uses `stream` in identifiers, strings, or comments. The workaround also couples dataset reference wrapping to reward-hack regex behavior.
- Fix approach: Replace text mutation with source review modes or AST/token-aware checks. If mutation remains, keep explicit tests for source text where `stream` appears in string literals, comments, and variable names.

## Known Bugs

**No concrete always-reproducing functional bug detected in the static scan:**
- Symptoms: Not detected.
- Files: `src/sol_execbench/`, `tests/sol_execbench/`
- Trigger: Not applicable.
- Workaround: Not applicable.

**ROCprof CSV parser can fail hard on malformed numeric fields:**
- Symptoms: `_duration_ns()` converts CSV duration/start/end fields with `float(value)` and does not catch `ValueError`.
- Files: `src/sol_execbench/core/bench/rocm_profiler.py`, `tests/sol_execbench/test_rocm_profiler.py`
- Trigger: A `rocprofv3` CSV row with a recognized duration column containing a non-numeric value.
- Workaround: Ensure profiler CSV inputs are generated by supported `rocprofv3` versions; add parser hardening before accepting external CSV fixtures.

## Security Considerations

**Untrusted solution code executes in a subprocess, not a sandbox:**
- Risk: Python and native HIP/C++ submissions are imported and executed with the current process permissions inside the evaluation subprocess. Static review blocks common file/process/network/dynamic-loader patterns, but this is not OS-level isolation.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/templates/build_ext.py`
- Current mitigation: Source paths reject absolute paths and `..` in `src/sol_execbench/core/data/solution.py`; Python submissions get static source review and blocked `torch.utils.cpp_extension.load/load_inline`; compilation/evaluation run in staging subprocesses with timeouts.
- Recommendations: Run untrusted submissions only inside the Docker/GPU isolation path; add OS-level resource limits and a restricted filesystem/network profile if evaluating third-party code outside trusted CI.

**Native compile flags are user-controlled:**
- Risk: `CompileOptions.cflags`, `hip_cflags`, and `ld_flags` are passed into `torch.utils.cpp_extension.load()` for native ROCm submissions.
- Files: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/driver/templates/build_ext.py`, `tests/sol_execbench/driver/test_build_ext.py`
- Current mitigation: Compilation occurs in a staging directory and solution source paths are constrained; legacy CUDA option keys are rejected.
- Recommendations: Treat native submissions as arbitrary code. If this runner is exposed to untrusted users, constrain compiler/linker flags to an allowlist or execute compilation in a disposable container with no sensitive mounts.

**Safetensors path resolution accepts paths outside staging when already absolute:**
- Risk: Workload safetensors inputs can resolve to absolute paths if present in workload data. The readiness layer blocks unsafe dataset paths, but `load_safetensors()` itself does not enforce root containment.
- Files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/dataset/readiness.py`, `tests/sol_execbench/test_dataset_inventory_readiness.py`
- Current mitigation: Dataset readiness tests cover paths outside the dataset root, and the eval driver prefers staging plus `FLASHINFER_TRACE_DIR` roots.
- Recommendations: Keep root-containment validation close to workload ingestion for all external data paths. Do not call `load_safetensors()` on untrusted workload JSON without the readiness checks.

**Reward-hack defense is regex and identity-snapshot based:**
- Risk: Static source review and function identity snapshots block known exploit families but are not a complete malicious-code defense.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Current mitigation: The driver snapshots critical functions before user import, blocks common dynamic loading/process/network/file patterns, rejects tensor subclasses, and checks for thread count increases.
- Recommendations: Treat `REWARD_HACK` checks as benchmark-integrity guardrails, not a security sandbox. Add focused tests for any newly discovered exploit pattern before broadening regexes.

## Performance Bottlenecks

**Correctness loop runs ten full input/reference/user rounds per workload:**
- Problem: Every workload runs multiple correctness rounds before timing, and each round may generate fresh tensors and synchronize GPU work.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`
- Cause: The loop is designed to catch nondeterminism and input-dependent errors.
- Improvement path: Keep the default rigorous path for benchmark runs; expose any faster mode only as clearly non-canonical, and ensure trace metadata distinguishes it.

**Timing intentionally synchronizes and clears cache around every measured iteration:**
- Problem: Device-event timing creates event arrays, clears an L2-sized cache buffer, synchronizes around each iteration, and uses `ShiftingMemoryPoolAllocator` copies.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/io.py`, `tests/sol_execbench/core/bench/test_timing.py`
- Cause: Benchmark semantics prioritize cold-cache timing and reward-hack resistance over raw throughput of the evaluator itself.
- Improvement path: Do not optimize these costs away without changing benchmark semantics. Add profiler-backed evidence in `src/sol_execbench/core/bench/rocm_profiler.py` when source-specific kernel timing is required.

**Dataset execution is serial at the problem level:**
- Problem: `scripts/run_dataset.py` invokes the CLI once per problem and waits for each subprocess before moving to the next.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/cli/main.py`
- Cause: GPU evaluation, compilation, clock locking, and output sidecars are easier to keep deterministic when serialized.
- Improvement path: If parallelism is added, shard by GPU or worker output root and make clock/profiler state explicit. Avoid sharing `out/`, timing evidence dirs, or staging dirs between concurrent runs.

## Fragile Areas

**Evaluation import order and fd redirection:**
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/problem_packager.py`
- Why fragile: The driver redirects stdout before importing torch/triton, imports benchmark helpers after staging setup, snapshots integrity before user import, and writes JSON traces to the saved stdout fd. Reordering these blocks can corrupt JSON output or weaken reward-hack checks.
- Safe modification: Preserve the order: redirect stdout, load problem/config, import helpers, static source review, exec reference, snapshot integrity, import user code, process workloads.
- Test coverage: Use `tests/sol_execbench/driver/test_eval_driver.py` for subprocess behavior and `tests/sol_execbench/test_e2e.py` for end-to-end traces.

**Source staging and cleanup lifecycle:**
- Files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/cli/main.py`
- Why fragile: `ProblemPackager.__del__()` removes the staging directory unless `keep_output_dir` is set. Destructor-based cleanup is sensitive to object lifetime and debugging workflows.
- Safe modification: Keep `--keep-staging` behavior intact and avoid adding long-lived references to `ProblemPackager` that change cleanup timing.
- Test coverage: `tests/sol_execbench/driver/test_problem_packager.py` covers staging behavior; end-to-end cleanup timing remains mostly integration-level.

**Hardware-gated tests can hide regressions on ordinary CI:**
- Files: `tests/conftest.py`, `tests/examples/test_examples.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/docker/dependencies/`
- Why fragile: ROCm, ROCm-dev, CK, rocWMMA, RDNA4, CDNA3, and `timing_serial` tests are skipped unless the environment matches. Default `pytest` does not exercise full GPU timing or native extension behavior.
- Safe modification: Keep CPU-safe contract tests for schema and helper logic, and run hardware-marked suites explicitly in ROCm CI before changing compile, timing, or example paths.
- Test coverage: Strong unit coverage exists, but full semantic coverage requires ROCm hardware and selected marker runs.

**Derived score eligibility depends on warnings and status strings:**
- Files: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/solar_derivation.py`
- Why fragile: Score eligibility is driven by string statuses such as `scored`, `degraded`, and `unscored`, plus warning prefixes like `unsupported_operator:*`.
- Safe modification: Add or change warning names only with tests that verify score eligibility, aggregate status, and serialized sidecar fields.
- Test coverage: `tests/sol_execbench/test_amd_native_score.py`, `tests/sol_execbench/test_amd_sol_v2.py`, and `tests/sol_execbench/test_solar_derivation_evidence.py` cover many cases but should be extended for every new operator family.

## Scaling Limits

**GPU memory and time scale with workload count and tensor sizes:**
- Current capacity: CLI defaults to `warmup_runs=10`, `iterations=50`, and an evaluation subprocess timeout of 600 seconds in `src/sol_execbench/cli/main.py`; dataset runner defaults to 300 seconds per problem in `scripts/run_dataset.py`.
- Limit: Large workloads, reference implementations, or native compilation can hit timeout or OOM even when the candidate kernel is correct.
- Scaling path: Use `--max-workloads`, `--limit`, `--iterations`, `--warmup-runs`, and ready-subset inputs for bounded runs; use Docker/ROCm CI for full validation.

**Derived sidecar generation grows with every traced workload:**
- Current capacity: AMD SOL v2 and SOLAR derivation sidecars are generated per workload when requested.
- Limit: Full dataset runs can produce many JSON sidecars and expensive graph/evidence derivations.
- Scaling path: Write sidecars under dedicated artifact dirs and use `_safe_sidecar_stem()` naming from `scripts/run_dataset.py`; keep artifact generation optional for exploratory runs.

**Profiler-backed timing depends on external `rocprofv3`:**
- Current capacity: `collect_source_timing_evidence()` can collect live CSV evidence when the selected timing policy uses `rocprofv3` and the tool is available.
- Limit: Missing profiler, command failures, or missing CSV output fall back to explicit non-profiler metadata.
- Scaling path: Keep fallback metadata visible in reports and run profiler evidence collection only where `rocprofv3` overhead and permissions are acceptable.

## Dependencies at Risk

**Pinned ROCm wheel stack:**
- Risk: `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, and `triton-rocm==3.6.0` are tightly coupled to ROCm wheel indexes and supported platform markers.
- Impact: Dependency resolution or runtime behavior can fail when ROCm wheel indexes change, when Python support moves, or when the local ROCm driver stack is mismatched.
- Migration plan: Update pins through `pyproject.toml` and `uv.lock` together; verify `tests/docker/dependencies/`, `tests/examples/test_examples.py`, and ROCm-marked timing/native tests.

**External ROCm command-line tools:**
- Risk: `rocm-smi`, `rocminfo`, `rocm_agent_enumerator`, `hipcc`, and `rocprofv3` are invoked by runtime, compile, diagnostics, and Docker checks.
- Impact: Missing tools degrade clock locking, architecture detection, native builds, diagnostics, or profiler evidence.
- Migration plan: Keep tool availability checks in `tests/docker/dependencies/`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/driver/problem_packager.py`, and `src/sol_execbench/core/bench/rocm_profiler.py` explicit and fail with actionable messages.

**Hugging Face dataset access path:**
- Risk: Dataset download depends on the `datasets` package and the external SOL ExecBench dataset source.
- Impact: Offline environments or upstream dataset changes can block local data refreshes.
- Migration plan: Preserve manifest/checksum generation in `scripts/download_solexecbench.py` and dataset inventory checks under `src/sol_execbench/core/dataset/`.

## Missing Critical Features

**OS-level sandboxing for untrusted evaluation:**
- Problem: The evaluator provides subprocess isolation and benchmark-integrity guardrails, but not a hardened sandbox.
- Blocks: Safe multi-tenant execution of arbitrary third-party Python/HIP submissions on hosts with sensitive files or network access.

**Automated full-matrix hardware validation in default test command:**
- Problem: The default pytest run skips many GPU, timing, and architecture-specific tests unless hardware and markers are selected.
- Blocks: Confidence that RDNA4 and CDNA3 behavior both remain valid after changes to timing, native compilation, examples, or hardware models.

**Direct ROCm API naming abstraction:**
- Problem: Timing still exposes compatibility names such as `bench_time_with_cuda_events()` and methodology values like `cuda_events`/`cupti`.
- Blocks: A fully ROCm-native public API surface; this is mostly naming debt, not runtime NVIDIA support.

## Test Coverage Gaps

**Security sandbox behavior is not OS-enforced:**
- What's not tested: Filesystem, network, process, and native dynamic-loading escape attempts beyond the known static review patterns.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: A new exploit pattern can bypass benchmark-integrity checks while still passing ordinary correctness tests.
- Priority: High

**Malformed profiler CSV and external tool oddities:**
- What's not tested: Non-numeric duration fields and unusual but syntactically valid `rocprofv3` CSV headers/rows.
- Files: `src/sol_execbench/core/bench/rocm_profiler.py`, `tests/sol_execbench/test_rocm_profiler.py`
- Risk: Timing evidence generation can fail abruptly instead of emitting explicit fallback metadata.
- Priority: Medium

**Full dataset and hardware matrix coverage:**
- What's not tested: Complete full-dataset execution across RDNA4 and CDNA3 in default local test runs.
- Files: `scripts/run_dataset.py`, `tests/conftest.py`, `tests/examples/test_examples.py`, `tests/docker/dependencies/`
- Risk: Changes can pass CPU-safe tests while breaking native HIP builds, ROCm library examples, clock locking, or device-event timing on one architecture.
- Priority: High

**Reference-source wrapping edge cases:**
- What's not tested: Source text where replacing `stream` with `strm` changes code semantics while building reference/custom solutions.
- Files: `scripts/run_dataset.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`, `tests/sol_execbench/test_run_dataset_amd_score.py`
- Risk: Dataset reference runs can produce misleading failures or altered benchmark behavior.
- Priority: Medium

---

*Concerns audit: 2026-05-24*
