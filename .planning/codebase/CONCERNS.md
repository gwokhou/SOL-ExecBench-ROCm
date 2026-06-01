# Codebase Concerns

**Analysis Date:** 2026-05-31
**v1.21 Update:** 2026-06-01

## v1.21 Closure Status

| Category | Status | Notes |
| --- | --- | --- |
| Dataset runner helper debt | Narrowed | Selection, run-state, closure, provenance, and derived-evidence refs now have package helpers and focused tests. The script remains the CLI orchestrator. |
| Eval driver template drift | Narrowed | Staged problem/reference/user import setup moved to `core.bench.eval_runtime`; timing/correctness/trace glue remains in the subprocess template. |
| AMD/SOLAR/static evidence monoliths | Narrowed | Classification, estimate dispatch, SOLAR status/boundary, and static extractor aggregate helpers are split out with focused tests. Parser/formula/extractor bodies remain substantial. |
| Reward-hack, clock, static evidence, dataset closure tests | Narrowed | Additional CPU-safe guardrails cover known bypass families, unsupported clock output, static aggregate states, and derived-evidence closure combinations. |
| Hard sandbox, multi-tenant safety, CDNA3/MI300X validation, paper-scale parity, hosted leaderboard readiness | Externally blocked/deferred | These require runner architecture, hardware evidence, paper-scale execution, policy, and operations work outside v1.21. |

## Tech Debt

**Large Monolithic Analysis Modules:**
- v1.21 status: Narrowed. Operation-family classification, estimate dispatch grouping, SOLAR status/source-boundary helpers, and static extractor aggregate status helpers are now separate modules with tests. Full parser/formula/extractor decomposition remains future work.
- Issue: The AMD/SOLAR scoring and graph derivation logic is concentrated in very large files with many tightly coupled helper functions and implicit semantic rules.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`
- Impact: Changes to one operation family, confidence rule, or evidence field can regress unrelated scoring paths. Review and test selection are harder because behavior is not isolated by operation family or evidence layer.
- Fix approach: Split by responsibility: graph extraction, operator-family classification, bound formulas, evidence serialization, and aggregate report rendering. Keep public schemas stable and move family-specific behavior behind small pure helpers with focused tests in `tests/sol_execbench/test_amd_bound_graph.py`, `tests/sol_execbench/test_amd_bound_estimates.py`, and `tests/sol_execbench/test_solar_derivation_evidence.py`.

**Dataset Runner Script Accretion:**
- v1.21 status: Narrowed. Dataset selection/run-state and closure/provenance helpers now live under `sol_execbench.core.dataset`; `scripts/run_dataset.py` still owns CLI orchestration and subprocess execution.
- Issue: `scripts/run_dataset.py` combines CLI argument parsing, dataset selection, execution, resume behavior, trace inspection, closure provenance, derived evidence, scoring, report writing, and output cleanup in one large script.
- Files: `scripts/run_dataset.py`
- Impact: Resume behavior and provenance handling are fragile because execution, filtering, skipping, and derived-report extension share mutable local state. A small change to `--max-workloads`, `--rerun`, or scoring flags can change closure semantics. Derived evidence reference construction is now isolated in `sol_execbench.core.dataset.evidence_refs`, but the main runner state machine remains monolithic.
- Fix approach: Continue moving reusable pieces into `src/sol_execbench/core/dataset/` modules: selection, run-state loading, closure record construction, derived evidence discovery, and output persistence. Keep `scripts/run_dataset.py` as orchestration only.

**Compatibility Naming Debt From CUDA-era APIs:**
- Issue: ROCm execution intentionally uses PyTorch's historical `torch.cuda` namespace and compatibility wrapper names such as `bench_time_with_cuda_events`.
- Files: `src/sol_execbench/core/bench/timing.py`, `tests/sol_execbench/core/bench/test_timing.py`, `docs/rocm_timing.md`, `tests/sol_execbench/test_rocm_migration_residue_audit.py`
- Impact: New contributors can misread compatibility APIs as NVIDIA runtime support. The audit test classifies accepted residue, but every new CUDA/NVIDIA term requires explicit classification to avoid ambiguity.
- Fix approach: Keep compatibility wrappers only at PyTorch API boundaries. Prefer ROCm-neutral names in new code (`device_events`, `hip`, `rocm`) and update `tests/sol_execbench/test_rocm_migration_residue_audit.py` whenever residue is intentional.

**ProblemPackager Staging Cleanup Still Has Destructor Fallback:**
- Issue: `ProblemPackager` now supports explicit `close()` and context-manager cleanup, but `__del__` remains as a best-effort fallback for callers that do not close explicitly.
- Files: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/cli/main.py`
- Impact: New callers can use deterministic cleanup, and the main CLI now closes the packager on normal completion and known `sys.exit(1)` error exits. Legacy callers and unhandled interpreter termination paths that rely on object collection still inherit Python object-lifetime timing. Staging directories can persist after abnormal termination.
- Fix approach: Prefer `with ProblemPackager(...)` or explicit `close()` in new call sites. Gradually migrate existing callers away from destructor reliance while preserving `keep_output_dir` behavior.

**Generated Driver Template Drift:**
- v1.21 status: Narrowed. Staged problem loading, reference loading, entry-point parsing, native ROCm detection, dynamic extension blocking, and user function loading now live in `sol_execbench.core.bench.eval_runtime`.
- Issue: Evaluation behavior lives in a copied template script rather than an importable runtime module.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/problem_packager.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Impact: Template edits are harder to type-check and share with the rest of the package. Runtime failures appear only after staging and subprocess execution. Shape/dtype structural output validation and output-call normalization are now delegated to importable helpers, but most evaluation flow still lives in the template.
- Fix approach: Continue thinning the template by moving small pure helpers under `src/sol_execbench/core/bench/` or `src/sol_execbench/driver/`. Add tests for each imported helper and keep template smoke tests for integration.

## Known Bugs

**Static Evidence Validation Does Not Prove Benchmark Correctness:**
- Symptoms: The recorded v1.17 RDNA 4 static-evidence run collected artifacts successfully, but all 14 workloads returned `RUNTIME_ERROR` with `hidden_states must be a HIP tensor`.
- Files: `docs/internal/v1_17_static_kernel_evidence_validation.md`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `examples/hip_cpp/rmsnorm/solution_hip.json`
- Trigger: Running the documented static-evidence command against `examples/hip_cpp/rmsnorm` on the recorded RDNA 4 environment.
- Workaround: Treat static evidence sidecars as diagnostic-only. Do not use them as correctness, timing, score, or leaderboard evidence until the benchmark run also passes.

**CDNA 3 / MI300X Hardware Validation Is Deferred:**
- Symptoms: The repo supports `gfx940`, `gfx941`, and `gfx942` metadata, but docs explicitly state no CDNA 3 or MI300X hardware-validation pass is recorded.
- Files: `docs/internal/cdna3_validation_readiness.md`, `docs/internal/mi300x_validation_readiness.md`, `docs/internal/non_cdna_validation_closure.md`, `tests/sol_execbench/test_rocm_test_suite_audit.py`
- Trigger: Any claim that CDNA 3, MI300X, `gfx942`, FP8-on-MI300X, or full adapted-suite validation is complete.
- Workaround: Keep public wording in readiness/deferred form and require a real `gfx94*` full-suite evidence run before changing support claims.

**Legacy Category Examples Are Compatibility Examples, Not Native Replacements:**
- Symptoms: Former NVIDIA categories such as CUTLASS, cuDNN, and cuTile remain represented by compatibility examples or migration documentation, while native ROCm replacements are separate paths.
- Files: `docs/rocm_libraries.md`, `docs/original_parity.md`, `examples/cutlass/gemm/`, `examples/cudnn/softmax/`, `examples/cutile/jamba_attn_proj/`, `tests/examples/test_examples.py`
- Trigger: Treating compatibility examples as proof of native CK, MIOpen, rocWMMA, or HIP parity for the original NVIDIA library category.
- Workaround: Use native examples under `examples/ck/`, `examples/miopen/`, `examples/rocwmma/`, `examples/hipblas/`, or `examples/hip_cpp/` for ROCm replacement evidence.

## Security Considerations

**Untrusted Solution Execution Boundary Is In-Process Python Plus Subprocess Isolation:**
- Risk: Submitted Python and native extension code is written to a staging directory and imported/executed by `eval_driver.py`. Static review blocks common file/network/process/dynamic-load patterns, but the evaluator is not a hard sandbox.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/problem_packager.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Current mitigation: Source paths reject absolute paths and `..`; static source review blocks known file I/O, subprocess, socket, loader, stream, cache, and precision-downgrade patterns; evaluation runs in a subprocess; dynamic `torch.utils.cpp_extension.load` is blocked for Python submissions.
- Recommendations: Run untrusted submissions only in disposable containers or isolated machines with no secrets mounted. Prefer OS-level sandboxing and filesystem/network restrictions over regex-only policy. Keep reward-hack tests for every new bypass family.

**Native Build Flags Are User-Controlled Inputs:**
- Risk: HIP/C++ solution metadata can pass compiler and linker flags into `torch.utils.cpp_extension`.
- Files: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/problem_packager.py`
- Current mitigation: Source paths are validated and legacy CUDA option keys are rejected. Native compilation is routed through declared ROCm language categories.
- Recommendations: Treat native builds as arbitrary code execution. If this is exposed beyond trusted local research workflows, constrain allowed `hip_cflags`, `cflags`, and `ld_flags`, and build inside a locked-down container without host-sensitive mounts.

**Docker Runtime Uses Broad GPU and Host-Mount Access:**
- Risk: `scripts/run_docker.sh` mounts the repository, passes `/dev/kfd` and `/dev/dri`, uses `--security-opt seccomp=unconfined`, `--ipc=host`, and grants passwordless sudo for ROCm SMI tools in the image.
- Files: `scripts/run_docker.sh`, `docker/Dockerfile`, `docker/entrypoint.sh`, `src/sol_execbench/core/bench/clock_lock.py`
- Current mitigation: The Docker flow is explicit and targeted at local GPU benchmarking. Clock commands use `sudo -n` and scoped SMI binaries.
- Recommendations: Do not run untrusted solutions in a container that mounts sensitive host paths. For shared systems, create a separate runner image/profile with restricted mounts, no interactive shell by default, and audited device permissions.

**Trace and Log Outputs Can Include Exception Text From User Code:**
- Risk: Runtime errors include exception messages and tracebacks in emitted trace logs.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/cli/main.py`, `scripts/run_dataset.py`
- Current mitigation: Secrets should not be mounted into benchmark environments, and `.env` files are not part of normal benchmark inputs.
- Recommendations: Treat trace JSONL, CLI logs, profiler output, and `.artifacts/` as potentially sensitive when evaluating untrusted submissions. Scrub paths and exception text before publishing external reports.

## Performance Bottlenecks

**Correctness Runs Ten Fresh Rounds Per Workload:**
- Problem: Every workload performs up to 10 correctness rounds before timing.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Cause: Multiple rounds catch nondeterministic and input-dependent issues, but multiply expensive reference and candidate execution costs.
- Improvement path: Keep the default rigorous path, but expose carefully documented developer-only knobs for fast local smoke tests. Do not use reduced rounds for score or claim evidence.

**Timing Clears L2 Cache and Allocates Unique Argument Pools:**
- Problem: Timing uses a shifting memory allocator and cache-clearing buffer across warmup and measured iterations.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/io.py`, `tests/sol_execbench/core/bench/test_timing.py`
- Cause: This protects benchmark semantics against pointer reuse and cache artifacts, but increases memory pressure and setup cost for large tensors.
- Improvement path: Keep current behavior for benchmark-grade runs. Add explicit memory diagnostics or preflight warnings for large workloads and document the relationship between `warmup`, `iterations`, and allocator footprint.

**Dataset Runs Recompute or Extend Derived Evidence Inline:**
- Problem: `scripts/run_dataset.py` extends AMD score, SOL bound, SOLAR derivation, timing evidence, and closure records while iterating problems.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/dataset/execution_closure.py`
- Cause: Derived reports are coupled to execution/resume decisions instead of a separate post-processing pipeline.
- Improvement path: Move derived evidence generation into resumable post-processing commands keyed by trace files and provenance, so dataset execution can finish independently.

**Docker Builds Reinstall ROCm Wheel Stack Per Target:**
- Problem: Docker image builds perform a frozen `uv sync` and then install target-specific PyTorch ROCm and Triton ROCm wheels.
- Files: `docker/Dockerfile`, `docker/rocm-targets.json`, `scripts/run_docker.sh`
- Cause: The checked-in lock targets the default ROCm stack while non-default Docker targets need different wheel versions.
- Improvement path: Preserve the matrix behavior, but document cache expectations and consider per-target lock artifacts if build reproducibility becomes more important than single-lock simplicity.

## Fragile Areas

**Reward-Hack Static Review Is Conservative Regex Matching:**
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Why fragile: Regex scanning can false-positive valid code and miss obfuscated behavior. Comment stripping is intentionally simple. Runtime integrity checks protect selected Python function identities but do not provide process-level confinement. Direct shell/process execution via common `os.*`, `getattr(os, ...)`, `__import__`, dynamic import, and `pty.spawn` APIs is now covered by focused static tests, but obfuscated process access remains possible.
- Safe modification: Add new blocked patterns only with tests showing both malicious and allowed cases. Prefer AST-based checks for Python when possible and document any intentional false positives.
- Test coverage: Good unit and driver coverage exists for known attacks; coverage is only as complete as the known attack catalog and available GPU runtime for driver-level smoke tests.

**Clock Locking Depends On Device Names, Sudoers, And ROCm SMI Output Text:**
- Files: `src/sol_execbench/core/bench/clock_lock.py`, `docker/entrypoint.sh`, `src/sol_execbench/core/bench/config/device_config.py`, `tests/sol_execbench/core/bench/test_clock_lock.py`
- Why fragile: Locking relies on passwordless SMI commands, text parsing, supported DPM levels, low-power-state interpretation, and device-name presets.
- Safe modification: Keep locking failure explicit when `lock_clocks=True`. Add device fixtures for every supported GPU family and include raw SMI output in validation artifacts.
- Test coverage: Unit tests cover parsing and negative paths; real hardware coverage is environment-dependent.

**Hardware-Sensitive Tests Are Frequently Skipped Outside ROCm Hosts:**
- Files: `tests/conftest.py`, `pyproject.toml`, `tests/docker/dependencies/`, `tests/sol_execbench/core/bench/test_timing.py`
- Why fragile: `requires_rocm`, `requires_rocm_dev`, `requires_ck`, `requires_rocwmma`, `requires_rdna4`, `requires_cdna3`, and `timing_serial` markers skip important behavior in non-GPU or non-target environments.
- Safe modification: Preserve marker semantics and add pure unit tests for parsing, schema, and report behavior when hardware is unavailable. Record real hardware validation separately.
- Test coverage: Non-hardware tests are broad, but timing validity, native builds, CK/rocWMMA/MIOpen behavior, and architecture-specific claims still require physical hardware or Docker with device passthrough.

**Public Claim Guardrails Are Text-Based:**
- Files: `tests/sol_execbench/test_v1_9_validation_closure.py`, `tests/sol_execbench/test_public_contract_guardrails.py`, `tests/sol_execbench/test_original_parity_docs.py`, `docs/CLAIMS.md`, `docs/analysis.md`
- Why fragile: Tests enforce forbidden/required phrases, which helps prevent overclaims but can miss semantically equivalent wording. Static Evidence and `rocprofv3` profile sidecar authority boundaries now have focused diagnostic-only guardrails, but new wording can still evade phrase-based checks.
- Safe modification: When changing claims docs, update tests with explicit allowed and forbidden language. Keep hardware-validation status in structured reports where possible.
- Test coverage: Good for known claim phrases and diagnostic-only sidecar authority wording; residual risk remains for new wording not covered by phrase guards.

**Static Kernel Evidence Is Diagnostic-Only And Toolchain-Sensitive:**
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/toolchain.py`, `docs/internal/v1_17_static_kernel_evidence_validation.md`, `tests/sol_execbench/test_static_kernel_evidence.py`
- Why fragile: Evidence depends on external ROCm tools (`llvm-objdump`, `readelf`, ROCm binary metadata), artifact availability, parser behavior, and bounded output capture.
- Safe modification: Treat missing or partial evidence as explicit status, not failure to parse silently. Add fixtures for every new artifact type and tool-run status.
- Test coverage: Contract tests exist for sidecars; real extractor coverage is bounded by installed toolchains and recorded validation runs.

## Scaling Limits

**Full Dataset Evaluation Requires Local Storage, Long Runtime, And GPU Access:**
- Current capacity: The package can run selected problems and capped workloads with `scripts/run_dataset.py --limit` / `--max-workloads`, and full runs expect downloaded benchmark assets under `data/`.
- Limit: Full benchmark-scale execution can be constrained by GPU availability, native build time, trace/profiler artifact size, and output directory growth.
- Scaling path: Keep capped smoke runs for development and use dedicated output roots for full runs. Add resumable post-processing and artifact retention policies before large multi-target validation.

**Architecture Validation Matrix Is Not Complete:**
- Current capacity: RDNA 4 has recorded validation artifacts; CDNA 3 readiness metadata exists; Docker targets cover ROCm versions through `docker/rocm-targets.json`.
- Limit: CDNA 3 / MI300X validation, CDNA 4 validation, FP8-on-MI300X, and NVFP4/MXFP4 AMD validation are explicitly deferred.
- Scaling path: Record separate hardware evidence entries per architecture, ROCm version, PyTorch ROCm wheel stack, and workload slice. Promote support claims only after evidence exists.

**Scoring Models Are Provisional For AMD-Native Interpretation:**
- Current capacity: AMD-native scoring emits guarded reports with confidence, warnings, baseline references, and derived evidence.
- Limit: AMD SOL/SOLAR-derived scores do not claim upstream SOLAR, NVIDIA B200, hosted leaderboard, or paper-scale parity.
- Scaling path: Keep score reports explicit about provenance and confidence. Add benchmark-calibrated hardware models and baselines per architecture before using scores for cross-system comparison.

## Dependencies at Risk

**Pinned PyTorch ROCm / Triton ROCm Stack:**
- Risk: Runtime dependencies pin `torch==2.10.0+rocm7.1`, `torchvision==0.25.0+rocm7.1`, and `triton-rocm==3.6.0` for Linux/Windows while the project baseline says ROCm >= 7.0.
- Impact: Users on ROCm 7.0 or non-default Docker targets can hit mixed-version or wheel availability issues.
- Migration plan: Keep `docker/rocm-targets.json` as the source of supported target stacks. Add target-specific compatibility tests and update `uv.lock` / Docker target metadata together.

**External ROCm Toolchain Availability:**
- Risk: `hipcc`, `rocprofv3`, `rocminfo`, `rocm-smi` / `amd-smi`, CK headers, rocWMMA headers, and MIOpen/hipBLAS libraries are environment-provided.
- Impact: Native examples, static evidence, profiler evidence, clock locking, and dependency reports degrade or skip depending on the host/container.
- Migration plan: Keep dependency preflight explicit through `src/sol_execbench/core/dependency_matrix.py`, `src/sol_execbench/core/diagnostics.py`, and `tests/docker/dependencies/`. Add new toolchain versions as matrix entries rather than implicit assumptions.

**Dataset Provider And Benchmark Asset Availability:**
- Risk: Dataset download and workload assets depend on external repositories and Hugging Face-style large artifacts.
- Impact: Full dataset testing and safetensors workload coverage cannot run from a clean checkout without downloads.
- Migration plan: Keep small synthetic fixtures under `tests/` for contracts, and treat `data/` as external. Pin dataset provenance in manifests and closure reports.

## Missing Critical Features

**Hard Sandbox For Untrusted Submissions:**
- Problem: The evaluator defends known reward hacks but does not provide a complete security sandbox.
- Blocks: Safe multi-tenant or adversarial public submission execution.

**Recorded CDNA 3 / MI300X Full-Suite Validation:**
- Problem: Readiness metadata exists, but no real `gfx94*` full-suite pass is recorded.
- Blocks: CDNA 3 hardware-validated claims and MI300X-specific benchmark confidence.

**Paper-Scale SOLAR / Leaderboard Equivalence:**
- Problem: AMD-native scoring and SOLAR derivation are guarded local interpretations, not upstream paper-scale or hosted leaderboard equivalence.
- Blocks: Claims comparing directly to NVIDIA B200, Blackwell, original SOLAR, or hosted leaderboard results.

**Native Replacement Coverage For All Former NVIDIA Library Categories:**
- Problem: Some former library categories remain compatibility examples or documented replacement directions rather than one-for-one native ROCm implementations.
- Blocks: Strong parity claims for CUTLASS, cuDNN frontend, CuTe DSL, and cuTile-style workloads.

## Test Coverage Gaps

**Real Hardware Timing And Clock Stability:**
- What's not tested: End-to-end timing validity with clocks locked across each supported GPU family and ROCm version.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/clock_lock.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/conftest.py`
- Risk: Timing results can be accepted from environments with unsupported clocks, unstable clocks, or architecture-specific event behavior.
- Priority: High

**CDNA 3 Native Execution:**
- What's not tested: Full adapted test suite and representative dataset runs on real `gfx94*` hardware.
- Files: `docs/internal/cdna3_validation_readiness.md`, `docs/internal/mi300x_validation_readiness.md`, `tests/conftest.py`, `src/sol_execbench/core/data/solution.py`
- Risk: Schema support and readiness reports can be mistaken for hardware validation.
- Priority: High

**Reward-Hack Bypass Families Beyond Current Catalog:**
- What's not tested: Obfuscated Python file/network/process access, native extension side effects, non-regex-detected stream misuse, and C++-level timing manipulation beyond known patterns.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: Malicious submissions can evade benchmark semantics or access the runner environment.
- Priority: High

**Static Evidence Across Toolchain Variants:**
- What's not tested: Static kernel evidence extraction across all declared Docker targets, architectures, and ROCm tool versions.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/toolchain.py`, `docker/rocm-targets.json`, `tests/sol_execbench/test_static_kernel_evidence.py`
- Risk: Sidecars can be partial, unavailable, or parser-sensitive on supported-looking environments.
- Priority: Medium

**Native ROCm Library Category Breadth:**
- What's not tested: Broad CK, rocWMMA, MIOpen, and hipBLAS workload coverage beyond small examples and dependency smoke tests.
- Files: `examples/ck/`, `examples/rocwmma/`, `examples/miopen/`, `examples/hipblas/`, `tests/sol_execbench/test_rocm_library_examples.py`, `tests/docker/dependencies/`
- Risk: Replacement-library readiness is overgeneralized from narrow examples.
- Priority: Medium

**Dataset Resume And Closure Provenance Combinations:**
- What's not tested: Large matrix of `scripts/run_dataset.py` combinations involving stale traces, stale closure provenance, capped workloads, ready subsets, reruns, and derived evidence options.
- Files: `scripts/run_dataset.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`, `src/sol_execbench/core/dataset/execution_closure.py`
- Risk: Reports can silently reuse stale outputs or mark filtered/unattempted workloads incorrectly.
- Priority: Medium

---

*Concerns audit: 2026-05-31*
