# Codebase Concerns

**Analysis Date:** 2026-06-04

## Tech Debt

**Dataset runner orchestration is large and stateful:**
- Issue: `scripts/run_dataset.py` is a 2000+ line command script that owns CLI invocation, resume/reuse decisions, ready-subset filtering, closure records, AMD score sidecars, SOLAR sidecars, timing sidecars, summary output, and phase routing in one module.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/runner.py`, `src/sol_execbench/core/dataset/execution_closure.py`
- Impact: Small changes to one concern can alter benchmark closure semantics, stale-output reuse, report provenance, or derived-evidence accounting. Regressions can look like valid skipped/reused results instead of runtime failures.
- Fix approach: Keep behavior changes tightly scoped and add tests in `tests/sol_execbench/test_run_dataset_execution_closure.py`, `tests/sol_execbench/test_run_dataset_amd_score.py`, and `tests/examples/test_rocm_cli_paths.py`. Move reusable pure helpers into `src/sol_execbench/core/dataset/runner.py` before adding more branches to `scripts/run_dataset.py`.

**AMD SOL and SOLAR derivation logic is concentrated in very large modules:**
- Issue: Bound graph extraction, work estimates, fallback estimation, and SOLAR evidence construction are implemented in large files with many local heuristics.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_sol.py`
- Impact: Operator-family changes can silently alter reported bound confidence, score eligibility, or evidence gaps. Fallback paths in `src/sol_execbench/core/scoring/amd_sol.py` intentionally keep deriving estimates after rich extraction fails, so failures may degrade evidence rather than stop execution.
- Fix approach: Treat scoring changes as contract changes. Add focused fixtures under `tests/sol_execbench/fixtures/solar_derivation/` and update tests such as `tests/sol_execbench/test_solar_derivation_evidence.py`, `tests/sol_execbench/test_amd_bound_estimates.py`, `tests/sol_execbench/test_amd_bound_graph.py`, and `tests/sol_execbench/test_amd_sol_bounds.py`.

**Legacy CUDA-named compatibility APIs remain intentional but fragile:**
- Issue: ROCm PyTorch exposes AMD devices through `torch.cuda`, so code still uses `torch.cuda.Event`, `torch.cuda.is_available()`, `torch.cuda.get_device_properties()`, and `cuda:0` device strings.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/conftest.py`, `docker/entrypoint.sh`, `src/sol_execbench/core/runtime_evidence.py`
- Impact: Removing CUDA-named strings mechanically would break ROCm execution. Leaving unreviewed CUDA/NVIDIA residue can also reintroduce unsupported CUDA semantics.
- Fix approach: Preserve PyTorch ROCm compatibility namespace usage only where covered by guardrail tests. Update `tests/sol_execbench/test_rocm_eval_timing_audit.py` and `tests/sol_execbench/test_rocm_migration_residue_audit.py` whenever changing runtime, timing, or diagnostics code.

**Static reward-hack review is regex/AST heuristic based:**
- Issue: `review_solution_sources()` blocks broad patterns for streams, file/process/network access, dynamic native loading, semantic caches, and precision downgrades.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Impact: False positives can reject valid submissions, and false negatives can compromise benchmark timing/correctness semantics. The checks are not a hard multi-tenant sandbox.
- Fix approach: Add malicious and benign samples before changing rules. Keep static review messages structured because `REWARD_HACK` traces are part of public behavior.

**Generated evaluation driver duplicates runtime boundaries:**
- Issue: The staged script `src/sol_execbench/driver/templates/eval_driver.py` imports many runtime helpers, redirects stdout, loads reference code, imports untrusted user code, performs reward-hack checks, and emits canonical JSONL.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/driver/problem_packager.py`
- Impact: Driver changes can break subprocess JSONL parsing, stderr/stdout separation, staging cleanup, reward-hack status, or trace schema behavior. CI excludes `tests/sol_execbench/driver/test_eval_driver.py` from the default GitHub workflow.
- Fix approach: Run `uv run pytest tests/sol_execbench/driver/test_eval_driver.py` for driver changes and add integration coverage when changing stdout redirection, module loading, or trace emission.

## Known Bugs

**No current deterministic code bug is recorded in source as an active FIXME:**
- Symptoms: The repository contains deferred validation and guardrail notes, but no actionable `FIXME`/`TODO` marker in `src/` that directly names a known broken implementation.
- Files: `src/sol_execbench/`, `tests/`, `docs/`
- Trigger: Not applicable.
- Workaround: Treat the fragile areas below as active risks even when tests pass.

**Timing fallback can hide profiler evidence loss:**
- Symptoms: Optional `rocprofv3` profiling falls back to normal evaluation when unavailable or failed; timing evidence collection records fallback metadata instead of failing the benchmark by default.
- Files: `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `docs/rocm_timing.md`
- Trigger: Missing `rocprofv3`, nonzero profiler exit, timeout, or missing CSV artifacts.
- Workaround: Inspect profile/timing sidecars and require `profiler_collected=true` or equivalent evidence before making kernel-activity timing claims.

## Security Considerations

**Evaluation executes submitted Python and native extensions locally:**
- Risk: The evaluation driver imports user-provided Python modules and compiled `benchmark_kernel.so` files in a subprocess. Static checks block many known exploit patterns, but this is not a hard sandbox.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/eval_runtime.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/problem_packager.py`
- Current mitigation: Static source review, `block_cpp_extension_load()`, reward-hack integrity checks, subprocess timeouts, staged execution, and stdout JSON isolation.
- Recommendations: Do not expose this runner as a hosted multi-tenant service without OS/container sandboxing, network/file isolation, device isolation, and resource limits. Keep docs aligned with the deferred hard-sandbox boundary in `docs/CLAIMS.md` and `docs/RESEARCHER-GUIDE.md`.

**Native HIP/C++ compilation runs repository-provided build scripts against submitted sources:**
- Risk: HIP/C++ solutions compile in a staging directory using `torch.utils.cpp_extension` and source-provided compile options.
- Files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/core/data/solution.py`
- Current mitigation: Solution schema rejects legacy CUDA language/options, auto-injects ROCm offload arches, and stages sources under a temporary directory.
- Recommendations: Review compile-option validation before allowing arbitrary external submissions. Treat `hip_cflags`, include paths, linked libraries, and native extension import as privileged local execution.

**Dataset redistribution boundaries are policy-sensitive:**
- Risk: NVIDIA SOL-ExecBench dataset rows, workloads, traces, solutions, blobs, and migrated derivatives are local-only and release-bundle-blocked.
- Files: `docs/provenance.md`, `provenance.toml`, `scripts/check_dataset_redistribution.py`, `scripts/check_prerelease_readiness.py`
- Current mitigation: Provenance policy, prerelease readiness checks, and redistribution guardrail scripts.
- Recommendations: Do not add dataset-derived fixtures under `tests/` or `docs/examples/` unless provenance policy explicitly allows them. Use synthetic fixtures for tests.

## Performance Bottlenecks

**GPU timing uses serialized event timing with cache clearing:**
- Problem: `time_runnable()` preallocates shifted arguments and uses PyTorch HIP-backed device events with synchronization around each measured iteration.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/io.py`, `tests/sol_execbench/core/bench/test_timing.py`
- Cause: Benchmark semantics require fresh data pointers and cold-cache behavior, and PyTorch ROCm event timing is the canonical fallback path.
- Improvement path: Do not remove synchronizations or cache clearing without benchmark-semantics review. Prefer optional diagnostic `rocprofv3` sidecars for kernel-activity evidence rather than changing canonical trace timing.

**Dataset-scale execution is mostly serial for ROCm runs:**
- Problem: `scripts/run_dataset.py` keeps the trace execution phase serial, while only the derived-only phase has `ThreadPoolExecutor` parallelism.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/runner.py`
- Cause: GPU evaluation, staging directories, device access, logs, and closure provenance are sensitive to concurrent execution.
- Improvement path: Add concurrency behind explicit scheduling controls, per-job output isolation, and closure-provenance tests. Avoid sharing a single GPU without explicit resource policy.

**Static evidence and profiler artifact collection can add substantial runtime and filesystem output:**
- Problem: Static kernel evidence and `rocprofv3` profiling create sidecars and tool artifacts around evaluation.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/cli/main.py`
- Cause: Tool invocation, artifact discovery, hashing, bounded output preservation, and profiler parsing happen outside the core trace path.
- Improvement path: Keep these features opt-in and diagnostic-only. Add timeout and artifact-size tests when expanding extractors.

## Fragile Areas

**Benchmark trace and closure semantics:**
- Files: `src/sol_execbench/core/data/trace.py`, `src/sol_execbench/core/data/contract.py`, `src/sol_execbench/core/dataset/execution_closure.py`, `src/sol_execbench/core/dataset/run_state.py`, `scripts/run_dataset.py`
- Why fragile: Trace status, closure status, skipped-existing-pass handling, provenance mismatch records, and derived-evidence gaps all feed researcher claims.
- Safe modification: Update contract guardrails in `tests/sol_execbench/test_public_contract_guardrails.py`, `tests/sol_execbench/test_execution_closure_contract.py`, and `tests/sol_execbench/test_dataset_run_state.py` with any schema or status change.
- Test coverage: Broad CPU tests exist, but dataset-scale GPU evidence still needs hardware runs for full confidence.

**Input generation heuristics affect correctness comparability:**
- Files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/bench/correctness.py`, `tests/sol_execbench/core/bench/test_io.py`, `tests/sol_execbench/core/bench/test_correctness.py`
- Why fragile: Heuristics generate masks, normalization weights, RoPE cos/sin tensors, SSM decays, softmax outputs, FP8, and FP4-like packed values. Small changes can make benchmark inputs easier, harder, or semantically different.
- Safe modification: Add tests for each affected input name/dtype/shape pattern. Keep random data generation deterministic through `set_seed()` in `src/sol_execbench/core/bench/correctness.py`.
- Test coverage: Unit coverage is strong for CPU-safe behavior; low-precision and GPU dtype behavior depends on ROCm/PyTorch support.

**ROCm hardware markers can skip important tests silently on non-ROCm hosts:**
- Files: `tests/conftest.py`, `pyproject.toml`, `.github/workflows/code-quality.yml`, `docs/TESTING.md`
- Why fragile: `requires_rocm`, `requires_rocm_dev`, `requires_ck`, `requires_rocwmma`, `requires_rdna4`, `requires_cdna3`, and `timing_serial` markers intentionally skip outside suitable hardware or explicit marker selection.
- Safe modification: When touching GPU behavior, run the relevant marked tests on a ROCm host with `/dev/kfd` and `/dev/dri`, and use `-m timing_serial -n 0` for timing-sensitive tests.
- Test coverage: GitHub Actions runs CPU-safe tests on Ubuntu for Python 3.12/3.13 and ignores driver E2E tests in `.github/workflows/code-quality.yml`.

**Docker target and dependency matrix can drift from Python dependency pins:**
- Files: `pyproject.toml`, `uv.lock`, `docker/Dockerfile`, `docker/rocm-targets.json`, `scripts/run_docker.sh`, `src/sol_execbench/core/dependency_matrix.py`
- Why fragile: The default project pins ROCm 7.1 PyTorch wheels, while Docker targets also model ROCm 7.0 and 7.2 explicit workflows. Non-default Docker images reinstall different PyTorch/Triton wheels after frozen sync.
- Safe modification: Update `docker/rocm-targets.json`, `docker/Dockerfile`, `pyproject.toml`, and dependency preflight tests together. Run `tests/sol_execbench/test_dependency_matrix_policy.py`, `tests/sol_execbench/test_run_docker_dependency_preflight.py`, and `tests/sol_execbench/test_docker_matrix_targets.py`.
- Test coverage: Preflight and script behavior are unit-tested, but actual image builds and GPU smoke checks require Docker and ROCm hardware.

**Claim-boundary wording is part of the product contract:**
- Files: `docs/CLAIMS.md`, `docs/RESEARCHER-GUIDE.md`, `docs/research_preview.md`, `scripts/build_prerelease_artifact_bundle.py`, `scripts/release_candidate_validation.py`, `src/sol_execbench/core/claim_upgrade.py`
- Why fragile: Reports must not upgrade diagnostic evidence into paper parity, upstream SOLAR parity, score authority, leaderboard readiness, hard sandboxing, native-host validation, CDNA4 validation, or full MI300X validation without explicit evidence.
- Safe modification: Run public docs and claim guardrail tests when editing evidence language: `tests/sol_execbench/test_public_contract_guardrails.py`, `tests/sol_execbench/test_research_release_docs.py`, and `tests/sol_execbench/test_prerelease_artifact_bundle.py`.
- Test coverage: Wording guardrails are extensive but can become brittle when legitimate terminology changes.

## Scaling Limits

**Full paper-scale validation remains evidence-heavy:**
- Current capacity: The code supports bounded slices, ready subsets, execution closure, per-problem traces, AMD-native reports, timing evidence, and paper denominator reports.
- Limit: Full 235-problem validation requires complete denominator accounting, trace artifacts, failure analysis, score artifacts, reproducible commands, and hardware evidence.
- Scaling path: Use `scripts/run_dataset.py` with explicit `--execution-closure`, `--ready-subset`, `--amd-score-report`, `--solar-derivation`, and `--timing-evidence-dir`; archive all sidecars and command provenance.

**Hardware validation is architecture- and host-specific:**
- Current capacity: RDNA 4 and CDNA 3 markers exist; docs record CDNA 3 `gfx942` adapted-suite evidence and MI300X benchmark-grade requirements.
- Limit: MI300X validation needs exact GPU identity, `gfx942`, clock-lock evidence, dataset traces, timing evidence, AMD-native score report, FP8 status, and deferred NVFP4/MXFP4 status.
- Scaling path: Follow `docs/internal/mi300x_validation_readiness.md` and `docs/internal/cdna3_validation_readiness.md`; do not treat marker readiness or Docker evidence as commercial GPU validation.

**Local cache/build/output directories can grow quickly:**
- Current capacity: Staging directories are temporary unless `--keep-staging` is set; dataset outputs and downloaded assets live under user-selected output dirs and `data/`.
- Limit: Full runs with traces, static evidence, profiler artifacts, derived sidecars, and retained staging dirs can consume significant disk.
- Scaling path: Keep generated datasets and outputs out of git; use bounded slices and cleanup output dirs before reruns when closure provenance permits.

## Dependencies at Risk

**ROCm PyTorch and Triton wheel availability is external and version-specific:**
- Risk: `pyproject.toml` pins `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, and `triton-rocm==3.6.0` for Linux x86_64; Docker target `rocm-7.2.0` declares PyTorch 2.11.0 ROCm 7.2 wheels.
- Impact: Dependency resolution, Docker builds, and GPU behavior can break if wheel indexes or compatibility shift.
- Migration plan: Update `pyproject.toml`, `uv.lock`, `docker/rocm-targets.json`, and dependency matrix tests in one change. Verify with `uv sync --all-groups`, Docker preflight tests, and a ROCm smoke run.

**ROCm command-line tools are host/container dependent:**
- Risk: `rocprofv3`, `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, and `rocm-smi` may be missing, renamed, or permission-gated.
- Impact: Profiling, environment evidence, offload-arch detection, clock locking, and Docker preflight can downgrade to unavailable/deferred states.
- Migration plan: Keep tool use routed through `src/sol_execbench/core/toolchain.py`, `src/sol_execbench/core/environment.py`, `src/sol_execbench/core/bench/clock_lock.py`, and `src/sol_execbench/core/bench/rocm_profiler.py`; add explicit unavailable tests.

**ROCm library examples depend on optional headers/libraries:**
- Risk: CK, rocWMMA, MIOpen, hipBLAS, and HIP development headers may not exist on every ROCm host.
- Impact: Example tests or native extension builds skip or fail depending on host packages.
- Migration plan: Preserve marker-based skips in `tests/conftest.py`, document prerequisites in `docs/rocm.md`, and add Docker dependency tests under `tests/docker/dependencies/` when adding library support.

## Missing Critical Features

**Hard sandboxing for hostile submissions:**
- Problem: The runner has reward-hack defenses but is not an isolation boundary for untrusted multi-tenant execution.
- Blocks: Hosted leaderboard, remote submission service, and adversarial public execution.

**Full 235-problem ROCm paper validation:**
- Problem: Current docs distinguish bounded preview evidence and denominator accounting from paper parity.
- Blocks: Paper-parity claims, leaderboard authority, and upstream SOLAR equivalence claims.

**Complete MI300X benchmark-grade validation:**
- Problem: CDNA 3 readiness and adapted-suite evidence are not the full MI300X evidence chain.
- Blocks: Commercial MI300X validation claims until `docs/internal/mi300x_validation_readiness.md` requirements are satisfied.

**RGA-rich static resource extraction and paper-scale static coverage:**
- Problem: Static kernel evidence is diagnostic-only and currently bounded to routed extractor/artifact sidecars.
- Blocks: Static evidence as performance, score, timing, or resource-usage authority.

## Test Coverage Gaps

**GPU driver E2E is not part of default CI:**
- What's not tested: Generated eval-driver execution, reward-hack subprocess behavior, native extension loading, and ROCm timing on actual devices.
- Files: `.github/workflows/code-quality.yml`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/test_e2e.py`
- Risk: CPU-safe CI can pass while GPU subprocess behavior regresses.
- Priority: High for changes under `src/sol_execbench/driver/`, `src/sol_execbench/core/bench/`, or `src/sol_execbench/cli/main.py`.

**Timing and profiler behavior needs real ROCm hardware:**
- What's not tested: Event timing stability, `rocprofv3` kernel activity CSV compatibility, clock lock effects, and non-default stream capture behavior across ROCm versions.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/clock_lock.py`, `tests/sol_execbench/core/bench/test_timing.py`
- Risk: Timing sidecars or canonical latencies can be misleading on unsupported hardware/tool combinations.
- Priority: High for timing, profiling, and scoring work.

**Docker and ROCm Matrix validation is mostly preflight-tested:**
- What's not tested: Real Docker image builds for every target and actual GPU execution inside each target.
- Files: `docker/Dockerfile`, `docker/rocm-targets.json`, `scripts/run_docker.sh`, `tests/sol_execbench/test_run_docker_matrix_script.py`
- Risk: Matrix entries can remain schema-valid while a target image or dependency stack breaks at runtime.
- Priority: Medium to High before release or compatibility claims.

**Low-precision and architecture-specific behavior needs target hardware:**
- What's not tested: FP8, FP4-like packed dtype behavior, CDNA3-specific behavior, RDNA4-specific behavior, CK/rocWMMA/MIOpen runtime behavior on representative GPUs.
- Files: `src/sol_execbench/core/bench/io.py`, `tests/sol_execbench/test_low_precision_compatibility.py`, `tests/sol_execbench/test_cdna3_hardware_marker.py`, `examples/ck/gemm/`, `examples/rocwmma/gemm/`, `examples/miopen/softmax/`
- Risk: Schema and CPU tests can pass while hardware support remains unavailable, skipped, or numerically unstable.
- Priority: Medium for ordinary code changes, High before hardware-validation claims.

---

*Concerns audit: 2026-06-04*
