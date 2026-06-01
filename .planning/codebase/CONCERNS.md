# Codebase Concerns

**Analysis Date:** 2026-06-01

## Tech Debt

**Generated evaluation driver owns too many benchmark responsibilities:**
- Issue: `src/sol_execbench/driver/templates/eval_driver.py` performs stdout redirection, source review, reference import, user import, input generation, ten-round correctness checks, reward-hack checks, timing, reference timing, trace emission, and device/clock validation in one generated script.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/timing.py`
- Impact: Benchmark behavior is difficult to change safely because security, correctness, timing, and trace semantics share one process-local namespace. Small changes can alter public trace behavior, reward-hack detection, or timing methodology.
- Fix approach: Keep `src/sol_execbench/driver/templates/eval_driver.py` as the thin generated entry point and move workload evaluation phases into importable functions under `src/sol_execbench/core/bench/` with focused tests under `tests/sol_execbench/driver/` and `tests/sol_execbench/core/bench/`.

**Large derived-scoring modules concentrate model logic and parser fallbacks:**
- Issue: `src/sol_execbench/core/scoring/solar_derivation.py` (~2560 lines), `src/sol_execbench/core/scoring/amd_bound_graph.py` (~1719 lines), and `src/sol_execbench/core/scoring/amd_bound_estimates.py` (~1455 lines) contain many operation-family branches, confidence classifications, and unsupported/degraded fallbacks.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`, `tests/sol_execbench/test_amd_bound_estimates.py`, `tests/sol_execbench/test_amd_bound_graph.py`
- Impact: Operation-estimate changes carry high regression risk and make it easy to accidentally upgrade an unsupported or inexact estimate into score authority.
- Fix approach: Split family-specific graph and estimate logic into modules such as `src/sol_execbench/core/scoring/families/gemm.py`, `attention.py`, `moe.py`, and `ssm.py`; keep aggregate claim-boundary checks in `src/sol_execbench/core/scoring/amd_bound_sanity.py`.

**Dataset batch orchestration is a large script with many responsibilities:**
- Issue: `scripts/run_dataset.py` (~1234 lines) handles CLI subprocess execution, skip/reuse policy, stale provenance detection, summary writing, derived score extension, timing evidence collection, and execution-closure records.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/dataset/execution_closure.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`
- Impact: Reuse decisions and closure metadata are coupled to file layout and CLI behavior; a narrow change can silently skip needed reruns or emit stale evidence references.
- Fix approach: Move reuse/provenance decisions and closure-record construction into `src/sol_execbench/core/dataset/` services with direct tests, leaving `scripts/run_dataset.py` as argument parsing and orchestration.

**ROCm port still carries compatibility naming and residue boundaries:**
- Issue: PyTorch ROCm uses CUDA-named APIs and the repo intentionally preserves upstream labels, but active code and tests must classify every CUDA/NVIDIA occurrence.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/utils.py`, `src/sol_execbench/core/environment.py`, `scripts/run_docker.sh`, `tests/sol_execbench/test_rocm_migration_residue_audit.py`, `docs/compliance.md`
- Impact: New contributors can confuse compatibility namespaces such as `torch.cuda` with real NVIDIA runtime support, or add unclassified residue that weakens ROCm-only claims.
- Fix approach: Preserve the residue audit in `tests/sol_execbench/test_rocm_migration_residue_audit.py`; any new CUDA/NVIDIA spelling must be either removed or classified with a precise reason and documentation path.

**Public examples include compatibility placeholders for former NVIDIA library categories:**
- Issue: Several former NVIDIA categories remain as PyTorch compatibility examples rather than ROCm-native library implementations.
- Files: `examples/cutile/jamba_attn_proj/solution_cutile.json`, `examples/cutlass/gemm/solution_cutlass.json`, `examples/cudnn/softmax/solution_cudnn.json`, `docs/rocm_libraries.md`, `docs/compliance.md`, `tests/examples/test_examples.py`
- Impact: Users can mistake compatibility examples for validated native CK, rocWMMA, MIOpen, or hipBLAS replacements.
- Fix approach: Keep compatibility wording explicit in `docs/rocm_libraries.md` and require compiled ROCm-native examples plus `tests/examples/` coverage before changing category readiness or claims.

## Known Bugs

**Python solution import path can execute unintended modules when paths collide:**
- Symptoms: `load_user_function()` transforms an entry path into a dotted module name and imports it after inserting the staging directory into `sys.path`; duplicate module names already in `sys.modules` can affect resolution if names collide with prior imports in the eval process.
- Files: `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Trigger: A submitted Python source path such as `main.py::run` or a package-like path whose module name matches an already imported module in the generated driver process.
- Workaround: Use unique source filenames in solution specs and add regression tests before changing `load_user_function()`. Prefer `importlib.util.spec_from_file_location()` with unique generated module names for staged user modules.

**No-trace evaluation failures lose partial stderr context unless verbose or hard failure paths expose it:**
- Symptoms: `src/sol_execbench/cli/main.py` treats nonzero evaluation with stdout as parseable and later reports "No traces produced"; stderr handling depends on verbose mode or specific failure branches.
- Files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/problem_packager.py`, `tests/sol_execbench/test_e2e.py`
- Trigger: Evaluation subprocess emits non-JSON stdout or library noise that starts with non-trace lines, exits nonzero, and does not produce valid trace JSON objects.
- Workaround: Run with `--verbose` and inspect staging logs when `--keep-staging` is enabled. Improve `convert_stdout_to_traces()` and CLI failure reporting to persist bounded stdout/stderr sidecars for all no-trace outcomes.

**Static source review is conservative but not a sandbox boundary:**
- Symptoms: `review_solution_sources()` blocks many AST-visible file/process/network/dynamic-loader patterns, but it is still a static allow/deny pass over submitted source text.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Trigger: Obfuscated Python or native behavior that avoids the exact AST and regex patterns, or compiled native code that performs behavior outside Python-visible checks.
- Workaround: Treat the generated evaluation subprocess and Docker boundary as the execution isolation layer. Add new exploit fixtures under `tests/sol_execbench/samples/` and assert `REWARD_HACK` in `tests/sol_execbench/driver/test_eval_driver.py` for every newly discovered pattern.

## Security Considerations

**Submitted benchmark code executes with local process permissions:**
- Risk: Python, Triton, and HIP/C++ submissions are written into a temporary staging directory and executed/imported by `python eval_driver.py`; native submissions compile with `torch.utils.cpp_extension.load()`.
- Files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/eval_runtime.py`
- Current mitigation: Source paths reject absolute paths and `..` in `src/sol_execbench/core/data/solution.py`; static review blocks common file/process/network/dynamic-load patterns in `src/sol_execbench/core/bench/reward_hack.py`; dynamic C++ extension loads inside user Python are patched off in `src/sol_execbench/core/bench/eval_runtime.py`; CLI subprocesses are timeout-bounded in `src/sol_execbench/cli/main.py`.
- Recommendations: Run untrusted submissions only inside the Docker/GPU isolation path from `scripts/run_docker.sh`. Add OS-level sandboxing or container-per-evaluation isolation before accepting arbitrary third-party submissions outside trusted research environments.

**Docker runtime grants broad GPU and host integration privileges:**
- Risk: `scripts/run_docker.sh` runs containers with `/dev/kfd`, `/dev/dri`, `--security-opt seccomp=unconfined`, `--ipc=host`, repository bind mount, and optional extra user-provided Docker args.
- Files: `scripts/run_docker.sh`, `docker/Dockerfile`, `docker/entrypoint.sh`, `tests/sol_execbench/test_run_docker_matrix_script.py`
- Current mitigation: Docker target and dependency preflight gate execution through `src/sol_execbench/core/docker_matrix.py`; unknown targets require explicit `--allow-unknown-target`; runtime evidence sidecars document validation status.
- Recommendations: Keep Docker runs restricted to trusted hosts and trusted command lines. Do not forward secrets or additional host mounts through `DOCKER_ARGS`; add a hardened non-interactive evaluation mode if the project needs multi-tenant execution.

**Compile flags are user-controlled for native solutions:**
- Risk: `CompileOptions.cflags`, `hip_cflags`, and `ld_flags` flow from solution JSON into `torch.utils.cpp_extension.load()` with minimal semantic restrictions.
- Files: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/problem_packager.py`
- Current mitigation: Solution schema validates source paths, rejects legacy CUDA schema fields, and restricts language mixing; compilation runs in the staging directory and is timeout-bounded by `src/sol_execbench/cli/main.py`.
- Recommendations: For untrusted workloads, maintain a denylist or allowlist for linker/compiler flags, block link-time paths outside staging, and add tests in `tests/sol_execbench/core/data/test_solution.py` for dangerous flags.

**Environment snapshots record visibility variables and tool output tails:**
- Risk: Optional evidence can include environment-derived device visibility values and bounded command output tails that may reveal host topology.
- Files: `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/runtime_evidence.py`, `src/sol_execbench/cli/main.py`, `docs/CONFIGURATION.md`
- Current mitigation: Collection is explicit, timeout-bounded, and scoped to ROCm diagnostics; no secret-like files were detected during this scan.
- Recommendations: Keep environment sidecars out of public artifacts unless reviewed. Avoid adding broad environment dumps; record only named, documented variables in `docs/CONFIGURATION.md`.

## Performance Bottlenecks

**Correctness performs ten full reference/user rounds per workload:**
- Problem: Each workload executes input generation, reference run, user run, and numerical comparison up to ten times before timing.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/correctness.py`, `src/sol_execbench/core/bench/io.py`
- Cause: Multiple rounds catch nondeterministic and input-dependent correctness failures, but expensive references and large safetensors workloads multiply total runtime.
- Improvement path: Keep the default rigorous path, but add a documented quick-check config in `src/sol_execbench/core/bench/config/benchmark_config.py` for local development and make traces include the configured correctness round count.

**Timing allocator preallocates warmup plus measurement copies:**
- Problem: `ShiftingMemoryPoolAllocator` receives `warmup + rep` iterations and produces unique data pointers for every timed run.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/io.py`, `tests/sol_execbench/core/bench/test_timing.py`
- Cause: Unique pointers prevent pointer-keyed output caching, but large input/output shapes and high iteration counts increase VRAM pressure.
- Improvement path: Add explicit VRAM estimation and clear error messages before allocation in `src/sol_execbench/core/bench/io.py`; keep reward-hack resistance as the priority over memory reuse.

**L2 cache clearing allocates a device buffer based on reported cache size:**
- Problem: `bench_time_with_device_events()` allocates a cache-clearing tensor sized to at least twice `L2_cache_size`.
- Files: `src/sol_execbench/core/bench/timing.py`, `tests/sol_execbench/core/bench/test_timing.py`
- Cause: Cache-clearing preserves benchmark semantics, but driver-reported cache sizes or constrained VRAM can turn timing setup into an OOM source.
- Improvement path: Add an upper bound or config-controlled cache clear policy with trace-visible methodology fields; test both default and constrained-memory paths.

**Profiler evidence parsing can become expensive and diagnostic-only:**
- Problem: `rocprofv3` collection runs a second profiled evaluation and parses profiler artifacts for timing evidence.
- Files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `scripts/run_dataset.py`, `docs/rocm_timing.md`
- Cause: Profiler-backed evidence is optional and diagnostic, but when enabled it adds subprocess runtime, artifact IO, and CSV parsing on top of canonical trace generation.
- Improvement path: Keep profiler collection opt-in, preserve fallback reasons in sidecars, and avoid using profiler-derived values as canonical score authority unless `src/sol_execbench/core/bench/timing_policy.py` selects a supported profiler-backed policy.

## Fragile Areas

**Benchmark integrity checks depend on process-local monkey-patch snapshots:**
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Why fragile: The integrity snapshot captures selected global function identities before user import; adding or renaming benchmark-critical helpers without updating `_CRITICAL_NAMES` can create a bypass.
- Safe modification: Any new correctness, timing, input, output, or trace helper used by the generated driver must be included in the snapshot or made inaccessible to user code. Add an evil sample in `tests/sol_execbench/samples/` when changing this boundary.
- Test coverage: Existing tests cover monkey-patching, thread injection, lazy outputs, and dynamic extension attempts; add tests for each new helper or exploit family.

**Trace JSONL is the public contract and excludes derived-report metadata:**
- Files: `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/test_public_contract_guardrails.py`
- Why fragile: Derived evidence fields, score authority, profiling metadata, and environment snapshots are intentionally sidecars. Adding trace keys can break downstream consumers and public contract guardrails.
- Safe modification: Keep canonical trace changes behind explicit schema discussions and update `tests/sol_execbench/test_public_contract_guardrails.py` before writing new trace fields.
- Test coverage: Guardrail tests assert canonical top-level keys and forbidden derived-report key space; maintain these as blocking tests.

**ROCm hardware validation claims are intentionally bounded:**
- Files: `docs/compliance.md`, `docs/CLAIMS.md`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_hardware_models.py`, `tests/sol_execbench/test_amd_native_score.py`, `tests/conftest.py`
- Why fragile: Schema includes `gfx940`, `gfx941`, and `gfx942`, but documentation states full CDNA 3 validation is deferred; scoring code emits CDNA3/no-validation warnings.
- Safe modification: Do not upgrade CDNA 3 support language or score authority without real `requires_cdna3` runs and updated evidence sidecars.
- Test coverage: Architecture markers in `tests/conftest.py` skip unavailable hardware, so CI without CDNA 3 cannot prove CDNA 3 runtime behavior.

**Static evidence and toolchain routing are availability-sensitive:**
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/toolchain.py`, `docs/static_kernel_evidence.md`, `docs/rocm_toolchain_routing.md`, `tests/sol_execbench/test_static_kernel_evidence.py`
- Why fragile: `llvm-objdump`, `readelf`, and ROCm tools may be unavailable, unsupported, timed out, or routed differently by artifact type and architecture.
- Safe modification: Treat `unavailable`, `unsupported`, `failed`, `partial`, and `timeout` statuses as non-authoritative. Preserve tool-run reason codes and raw-output sidecars.
- Test coverage: Unit tests cover routing and sidecar shapes; real tool availability varies by machine and Docker target.

**Docker target matrix controls whether benchmark claims are allowed:**
- Files: `docker/rocm-targets.json`, `src/sol_execbench/core/docker_matrix.py`, `scripts/run_docker.sh`, `tests/sol_execbench/test_docker_matrix_targets.py`, `tests/sol_execbench/test_run_docker_matrix_script.py`
- Why fragile: Unknown targets and mixed dependency versions can still be requested explicitly, but they change claim authority and validation status.
- Safe modification: Add new Docker targets only through `docker/rocm-targets.json`, preserve preflight classifications, and require tests for preview, build args, runtime gating, and sidecar output.
- Test coverage: Script tests use dry-run/preflight fixtures; live Docker/GPU validation still requires `./scripts/run_docker.sh --record-container-validation`.

## Scaling Limits

**Single-process per-problem evaluation model:**
- Current capacity: One CLI invocation packages one definition, workload list, and solution into one staging directory, then one eval subprocess processes workloads serially.
- Limit: Large workload lists, expensive references, or memory-heavy kernels can produce long single-process runs and OOM risk.
- Scaling path: Add workload sharding in `scripts/run_dataset.py` and core helpers under `src/sol_execbench/core/dataset/runner.py`; keep one trace file per shard with deterministic merge rules.

**Dataset artifacts and safetensors live on local filesystem paths:**
- Current capacity: `FLASHINFER_TRACE_DIR` and `data/` are local paths mounted into Docker and searched by the eval driver.
- Limit: Missing datasets skip or fail workloads; large local data increases IO pressure and complicates reproducibility across hosts.
- Scaling path: Keep dataset provenance in `src/sol_execbench/core/dataset/manifest.py` and checksums in `src/sol_execbench/core/dataset/checksums.py`; add remote cache/index support only with checksum enforcement.

**Pytest defaults to parallel execution while GPU timing is serial-sensitive:**
- Current capacity: `pyproject.toml` uses `pytest -n auto --dist loadgroup`; `tests/conftest.py` skips `timing_serial` unless explicitly selected.
- Limit: GPU runtime tests can interfere with each other if new tests miss markers, and default CI can pass without exercising real ROCm hardware.
- Scaling path: Mark every GPU test with `requires_rocm` plus architecture/tool markers, and mark timing-sensitive tests with `timing_serial`.

## Dependencies at Risk

**Pinned ROCm/PyTorch/Triton stack:**
- Risk: `pyproject.toml` pins `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, and `triton-rocm==3.6.0`; `docker/Dockerfile` and `docker/rocm-targets.json` mirror this target.
- Impact: Wheel availability, ROCm minor-version drift, or PyTorch ROCm API changes can block installs or invalidate timing/build assumptions.
- Migration plan: Add declared Docker targets in `docker/rocm-targets.json`, update `uv.lock`, run dependency preflight through `src/sol_execbench/core/dependency_matrix.py`, and validate with `tests/docker/dependencies/`.

**PyTorch extension API still exposes CUDA-named keywords for HIP builds:**
- Risk: `src/sol_execbench/driver/templates/build_ext.py` uses `extra_cuda_cflags` and PyTorch CUDA namespace compatibility for HIP builds.
- Impact: Upstream PyTorch API changes can break native HIP/C++ compilation even when ROCm itself is installed.
- Migration plan: Keep build-path tests in `tests/sol_execbench/driver/test_build_ext.py` and `tests/sol_execbench/test_rocm_schema_build_audit.py`; isolate compatibility naming in the build template.

**ROCm command-line tools are machine-dependent:**
- Risk: `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi`, `rocprofv3`, `llvm-objdump`, and `readelf` are probed dynamically and may be missing or version-skewed.
- Impact: Environment snapshots, clock locking, offload-arch injection, profiling, and static evidence can degrade or become unavailable.
- Migration plan: Keep all tool invocations timeout-bounded and status-coded in `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/clock_lock.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, and `src/sol_execbench/core/bench/static_kernel_evidence.py`.

**Third-party dataset source and downloaded assets are external:**
- Risk: Dataset download scripts reference external Hugging Face/upstream assets, and downloaded benchmark assets are intentionally not committed.
- Impact: Dataset availability or upstream shape changes can break batch runs outside unit fixtures.
- Migration plan: Preserve local manifest/checksum validation in `src/sol_execbench/core/dataset/manifest.py`, `src/sol_execbench/core/dataset/checksums.py`, `scripts/download_solexecbench.py`, and tests such as `tests/sol_execbench/test_download_solexecbench.py`.

## Missing Critical Features

**Full CDNA 3 validation evidence:**
- Problem: `docs/compliance.md` states CDNA 3 full-suite validation is deferred even though `gfx94*` values are present in schema.
- Blocks: Authoritative CDNA 3 support claims, CDNA 3 score claims, and broad hardware compatibility claims.

**Native ROCm replacements for every former NVIDIA library category:**
- Problem: Some public categories remain compatibility examples rather than validated native CK, rocWMMA, MIOpen, hipBLAS, or HIP/Triton implementations.
- Blocks: Strong claims that all original SOL ExecBench library categories have equivalent AMD-native implementations.

**OS/container sandbox boundary for arbitrary untrusted submissions:**
- Problem: Static review plus subprocess execution is not a full sandbox for hostile code.
- Blocks: Safe multi-tenant hosted benchmark service or public arbitrary-code execution.

**Authoritative profiler-backed timing for all source types:**
- Problem: `docs/rocm_timing.md` documents unsupported, fallback, and diagnostic-only timing states for mixed or unknown sources.
- Blocks: Using profiler artifacts as canonical score authority across all solution categories.

## Test Coverage Gaps

**Real GPU coverage depends on local hardware and markers:**
- What's not tested: Default test runs can skip `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and `timing_serial` coverage when hardware or explicit marker selection is unavailable.
- Files: `tests/conftest.py`, `pyproject.toml`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/test_e2e.py`, `tests/docker/dependencies/`
- Risk: CPU-only CI can pass while ROCm driver, device-node, timing, and native extension behavior regresses.
- Priority: High

**Native HIP/C++ and ROCm library examples need live toolchain validation:**
- What's not tested: Unit tests verify schemas and dry-run behavior, but compiled CK, rocWMMA, hipBLAS, MIOpen, and HIP examples require ROCm headers, libraries, and GPU access.
- Files: `examples/hip_cpp/`, `examples/hipblas/`, `examples/ck/`, `examples/rocwmma/`, `examples/miopen/`, `tests/examples/test_examples.py`, `tests/docker/dependencies/`
- Risk: Example metadata can remain valid while real compilation or runtime behavior breaks.
- Priority: High

**Reward-hack defenses need continuous adversarial fixtures:**
- What's not tested: Unknown obfuscation techniques, native-code side effects, and future PyTorch/Triton execution APIs beyond the current evil samples.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/core/bench/eval_runtime.py`, `tests/sol_execbench/samples/`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: A submitted solution can evade detection and corrupt timing or correctness authority.
- Priority: High

**Dataset reuse and derived evidence paths need more failure-mode tests:**
- What's not tested: Many combinations of stale provenance, selected ready-subset workloads, missing derived sidecars, and rerun flags are concentrated in `scripts/run_dataset.py`.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/run_closure.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`, `tests/sol_execbench/test_dataset_run_closure.py`
- Risk: Batch runs can skip needed reevaluation or report incomplete closure status.
- Priority: Medium

**Static evidence tools need cross-machine integration tests:**
- What's not tested: Real `llvm-objdump`, `readelf`, RGA-planned routes, and artifact parser behavior across ROCm/Docker targets.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/toolchain.py`, `tests/sol_execbench/test_static_kernel_evidence.py`, `docs/rocm_toolchain_routing.md`
- Risk: Static evidence status can degrade silently on machines with different tool availability.
- Priority: Medium

---

*Concerns audit: 2026-06-01*
