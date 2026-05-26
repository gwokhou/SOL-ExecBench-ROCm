# Codebase Concerns

**Analysis Date:** 2026-05-26

## Tech Debt

**Large scoring derivation modules:**
- Issue: AMD SOL scoring is concentrated in very large modules that mix parsing, graph inference, work estimation, aggregation, warnings, and schema parsing.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/amd_score.py`
- Impact: New operation families, hardware models, or scoring warnings are easy to add inconsistently because validation, inference, and reporting logic are not isolated by concern.
- Fix approach: Keep public artifact schemas stable, but split internal helpers by responsibility: graph extraction in `src/sol_execbench/core/scoring/amd_bound_graph.py`, per-family estimates in modules under `src/sol_execbench/core/scoring/`, and artifact parsing/serialization in thin contract modules.

**Dataset runner is an orchestration monolith:**
- Issue: `scripts/run_dataset.py` owns CLI command construction, subprocess execution, filtering, closure records, score sidecar generation, timing evidence, summary writing, and rerun policy in one script.
- Files: `scripts/run_dataset.py`
- Impact: Changes to one execution mode can regress other modes, especially ready-subset filtering, skipped-existing traces, and derived evidence references.
- Fix approach: Move reusable runner pieces into package code under `src/sol_execbench/core/dataset/` or `src/sol_execbench/core/reporting.py`, then keep `scripts/run_dataset.py` as argument parsing plus top-level orchestration.

**Legacy CUDA naming remains in ROCm compatibility paths:**
- Issue: ROCm execution intentionally uses PyTorch's `torch.cuda` compatibility namespace, but CUDA/NVIDIA terms remain in API names, examples, tests, and sample filenames.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/core/bench/test_timing.py`, `tests/sol_execbench/samples/flux_rope/solution_cuda.json`, `tests/sol_execbench/samples/rmsnorm/solution_cuda.json`, `docs/rocm_timing.md`
- Impact: Maintainers can confuse intentional PyTorch ROCm compatibility with unsupported CUDA runtime support, and new tests may whitelist residue too broadly.
- Fix approach: Preserve compatibility call sites where PyTorch requires `torch.cuda`, but name local abstractions with device/HIP terminology and add explicit comments/tests when CUDA spelling is intentionally retained.

**Static source review is regex-based:**
- Issue: Reward-hack static review uses regular expressions over comment-stripped source text and blocks broad patterns like file I/O, subprocess access, streams, caching, and precision casts.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`
- Impact: The detector can both miss obfuscated Python/native behavior and block legitimate kernels whose source contains matching identifiers or safe helper code.
- Fix approach: Keep the conservative blocklist for benchmark execution, but add AST-based Python checks for import/call patterns and separate native-source checks by language so exceptions can be reviewed without weakening all rules.

**Hardware model validation status is hard-coded around gfx1200:**
- Issue: Hardware model parsing only allows validated status for `gfx1200`; CDNA 3 architectures are forced to provisional/unvalidated status even though project constraints include CDNA 3.
- Files: `src/sol_execbench/core/scoring/amd_hardware_models.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/data/amd_hardware_models/`
- Impact: CDNA 3 scoring remains degraded by construction until the hard-coded validation gate and packaged model payloads are updated.
- Fix approach: Replace `VALIDATED_GFX1200_ONLY` with a data-driven allowlist in hardware model JSON and tests that validate `gfx940`, `gfx941`, and `gfx942` independently.

## Known Bugs

**Static extractor timeout is reported as failed rather than timeout aggregate:**
- Symptoms: Extractor timeouts become `FAILED` tool runs and can collapse aggregate evidence to a generic failure instead of a distinct timeout status.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`
- Trigger: Run static extraction with `llvm-objdump` or `readelf` exceeding `timeout_seconds`.
- Workaround: Inspect `reason_code` for `extractor_timeout` in the static evidence sidecar.

**Reference latency failures are silently ignored:**
- Symptoms: When `benchmark_reference=True`, reference timing exceptions are swallowed and `reference_latency_ms` remains `0.0`, making `speedup_factor` `0.0` without an explicit trace warning.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`
- Trigger: Reference function passes correctness but fails during the separate timing call.
- Workaround: Treat zero reference latency as missing baseline evidence and prefer explicit scoring baseline artifacts from `src/sol_execbench/core/scoring/baseline_artifact.py`.

**Readiness NVIDIA blocker detection is shallow:**
- Symptoms: Dataset readiness only detects a small tuple of NVIDIA runtime hints such as `cupy`, `cuda.c`, `nvrtc`, `cublas`, and `cutlass`.
- Files: `src/sol_execbench/core/dataset/inventory.py`, `src/sol_execbench/core/dataset/readiness.py`
- Trigger: Canonical references contain other CUDA-only names or indirect imports not listed in `NVIDIA_RUNTIME_HINTS`.
- Workaround: Run execution closure and static review before making readiness claims; do not rely on inventory readiness alone for full ROCm compatibility.

## Security Considerations

**Submitted code executes inside a Python subprocess with GPU and filesystem access:**
- Risk: User solution and reference code are imported/executed from the staging directory; Python submissions can run arbitrary module import side effects before workload timing.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/core/data/solution.py`
- Current mitigation: Source paths reject absolute paths and `..`; static review blocks many Python process/network/file patterns; dynamic extension loading is patched out for Python submissions.
- Recommendations: Run benchmark execution in a constrained container/user namespace, keep staging directories isolated, restrict network access during evaluation, and treat static source review as a policy layer rather than a sandbox.

**Native HIP/C++ compile options are user-controlled:**
- Risk: `hip_cflags`, `cflags`, and `ld_flags` from solution metadata flow into the build template for native extensions.
- Files: `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/driver/templates/build_ext.py`, `src/sol_execbench/driver/problem_packager.py`
- Current mitigation: Solution path traversal is rejected and CUDA-specific compile option names are rejected; native compilation is separated from Python runtime loading.
- Recommendations: Add an allowlist or denylist for linker/compiler flags that can load unexpected libraries, write outside the build tree, or change runtime search paths.

**Docker image grants passwordless GPU clock tooling:**
- Risk: Container users receive passwordless `amd-smi` or `rocm-smi` access for clock control.
- Files: `docker/Dockerfile`, `src/sol_execbench/core/bench/clock_lock.py`, `scripts/run_docker.sh`
- Current mitigation: Clock commands are narrowly used for ROCm clock locking and verification; `scripts/run_docker.sh` guards against Docker Desktop contexts.
- Recommendations: Keep sudoers entries limited to exact tool paths, document host-level impact, and avoid expanding passwordless sudo beyond GPU clock tooling.

**Safetensors path resolution can load first matching partial-overlap path:**
- Risk: `_resolve_blob_path` progressively strips leading path components and accepts the first existing match under configured blob roots.
- Files: `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/core/dataset/readiness.py`
- Current mitigation: Dataset readiness rejects absolute safetensors refs and refs outside the dataset root; runtime roots are limited to staging and `FLASHINFER_TRACE_DIR`.
- Recommendations: Prefer manifest-backed exact asset paths for release runs and record resolved safetensors paths in trace/evidence sidecars for auditability.

## Performance Bottlenecks

**Timing allocator can allocate large per-workload pools:**
- Problem: `ShiftingMemoryPoolAllocator` preallocates pools for every input and DPS output across `warmup + rep` iterations, with offset padding per iteration.
- Files: `src/sol_execbench/core/bench/timing.py`, `src/sol_execbench/core/bench/io.py`, `src/sol_execbench/driver/templates/eval_driver.py`
- Cause: The allocator intentionally changes `data_ptr` per iteration to reduce cache/keyed-output exploits.
- Improvement path: Keep the anti-cheat property, but add memory budgeting and clearer errors when pool allocation would exceed available VRAM.

**Correctness loop regenerates and executes ten rounds per workload:**
- Problem: Every workload runs ten correctness rounds before timing, including reference execution and user execution each round.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`
- Cause: Multiple rounds catch nondeterminism and input-dependent correctness bugs.
- Improvement path: Preserve ten rounds for scored/release validation; allow clearly labeled quick modes for developer iteration that cannot produce benchmark claims.

**Dataset execution invokes one CLI subprocess per problem:**
- Problem: Dataset runs pay process startup, import, package staging, and optional compile overhead for each problem.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/cli/main.py`
- Cause: Isolation is problem-level rather than suite-level.
- Improvement path: Keep subprocess isolation for untrusted solutions, but cache validated staging/build artifacts by solution hash and add structured retry/resume around `scripts/run_dataset.py`.

## Fragile Areas

**Reward-hack defenses are timing-critical and bypass-sensitive:**
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Why fragile: Integrity snapshots, monkey-patch detection, stream blocking, lazy-output checks, and thread counts depend on import order and specific Python/PyTorch behavior.
- Safe modification: Change defenses only with adversarial regression tests in `tests/sol_execbench/driver/test_eval_driver.py` and `tests/sol_execbench/core/bench/test_reward_hack.py`.
- Test coverage: Good coverage exists for known attacks, but obfuscated imports, native extension side effects, and non-thread async work need ongoing adversarial tests.

**ROCm hardware tests are skipped by environment probes:**
- Files: `tests/conftest.py`, `docs/TESTING.md`, `tests/examples/test_examples.py`, `tests/sol_execbench/test_e2e.py`
- Why fragile: Missing `/dev/kfd`, `/dev/dri`, ROCm dev headers, CK headers, or rocWMMA headers silently converts important coverage to skips.
- Safe modification: Always report pass/skip/fail counts for hardware validation, and run `requires_rocm`, `requires_rocm_dev`, `requires_rdna4`, and `requires_cdna3` jobs on matching hardware before release claims.
- Test coverage: CPU-only CI can validate contracts but cannot validate GPU timing, native HIP extension behavior, or architecture-specific examples.

**Clock locking depends on host privileges and text parsing:**
- Files: `src/sol_execbench/core/bench/clock_lock.py`, `docker/Dockerfile`, `docs/rocm.md`
- Why fragile: `sudo -n rocm-smi` must work, DPM levels must match the device, and `_level_is_active` parses command output text.
- Safe modification: Add hardware-backed tests or captured-output fixtures whenever changing parsing or preset logic.
- Test coverage: Unit tests cover command construction/parsing, but real clock stability depends on the host GPU, driver, and container privileges.

**Derived AMD score claims rely on layered evidence discipline:**
- Files: `src/sol_execbench/core/scoring/amd_score.py`, `src/sol_execbench/core/scoring/amd_sol_v2.py`, `src/sol_execbench/core/scoring/solar_derivation.py`, `docs/CLAIMS.md`
- Why fragile: Scores can be numerically computed even when evidence is degraded, provisional, or unvalidated; warnings are the main guardrail preventing overclaiming.
- Safe modification: Preserve warning propagation and claim-level fields whenever changing scoring formulas or artifact parsers.
- Test coverage: Contract tests exist, but release claims still require artifact review against `docs/CLAIMS.md`.

## Scaling Limits

**Full benchmark validation is hardware-bound:**
- Current capacity: The public dataset target is full SOL ExecBench scale, while local commands commonly run bounded subsets via `scripts/run_dataset.py --limit`.
- Limit: Full validation requires ROCm GPUs, device passthrough, optional static tooling, and long-running subprocess execution.
- Scaling path: Use ready-subset manifests and execution closure sidecars to partition runs, then merge summaries and evidence by problem/workload IDs.

**Static evidence extraction is bounded but serial per artifact/tool:**
- Current capacity: Static extractors run with per-tool timeouts and bounded output tails.
- Limit: Large binaries or missing tools produce partial/failed evidence and slow suite-level collection.
- Scaling path: Parallelize extraction at the orchestrator level and persist route decisions so unavailable toolchains are not reprobed repeatedly.

**Packaged hardware model validation does not cover all supported architectures equally:**
- Current capacity: `gfx1200` can be marked validated, while CDNA 3 remains provisional/unvalidated in the parser contract.
- Limit: CDNA 3 score reports remain degraded and cannot support full validation claims.
- Scaling path: Add validated hardware model payloads and evidence refs for `gfx940`, `gfx941`, and `gfx942`.

## Dependencies at Risk

**ROCm/PyTorch compatibility namespace:**
- Risk: The project depends on PyTorch ROCm exposing AMD devices through `torch.cuda` APIs.
- Impact: Timing, device detection, stream checks, and tests break if PyTorch changes compatibility behavior.
- Migration plan: Keep a thin device-events abstraction in `src/sol_execbench/core/bench/timing.py` and isolate direct `torch.cuda` calls behind ROCm-named helpers.

**ROCprofiler and ROCm toolchain binaries:**
- Risk: `rocprofv3`, `llvm-objdump`, `readelf`, `rocm_agent_enumerator`, `rocminfo`, `rocm-smi`, and `amd-smi` availability varies by ROCm installation/container.
- Impact: Timing evidence, static evidence, target detection, and clock locking degrade or fail.
- Migration plan: Continue using `src/sol_execbench/core/toolchain.py` route decisions and include explicit unavailable/partial statuses in evidence sidecars.

**Hugging Face datasets and external benchmark assets:**
- Risk: Dataset download depends on `nvidia/SOL-ExecBench` and `flashinfer-ai/flashinfer-trace` availability/revisions.
- Impact: Reproducibility and readiness denominators can drift without pinned revisions and manifests.
- Migration plan: Require dataset manifests from `src/sol_execbench/core/dataset/manifest.py` for release evidence and pin revisions in scripted runs.

## Missing Critical Features

**Real CDNA 3 validation closure:**
- Problem: The codebase supports CDNA 3 markers and hardware enum values, but scoring hardware model validation is not complete for CDNA 3.
- Blocks: Full CDNA 3 release claims and validated AMD-native score claims.

**Strict sandboxing for submitted Python/native code:**
- Problem: Policy checks exist, but execution is still Python/native code in a subprocess with inherited environment and filesystem access.
- Blocks: Safe operation on untrusted submissions outside a controlled container.

**Complete ROCm replacement coverage for former NVIDIA library categories:**
- Problem: Some former NVIDIA categories are compatibility examples or candidate replacement paths rather than full native replacements.
- Blocks: Claims of complete parity across CUTLASS/cuDNN/cuTile-era benchmark categories.

## Test Coverage Gaps

**GPU behavior depends on hardware-specific skipped tests:**
- What's not tested: HIP extension compilation, device event timing, ROCm clock locking, RDNA 4 examples, CDNA 3 behavior, CK, and rocWMMA in CPU-only environments.
- Files: `tests/conftest.py`, `tests/examples/test_examples.py`, `tests/sol_execbench/test_e2e.py`, `tests/docker/dependencies/`
- Risk: Contract tests can pass while actual GPU execution is skipped.
- Priority: High

**Reward-hack coverage is adversarial but finite:**
- What's not tested: Every obfuscation variant for dynamic imports, network/process access, native side effects, async GPU work outside Python thread counts, and source patterns that bypass regex matching.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`
- Risk: A malicious or over-optimized solution can evade detection and contaminate timing or correctness results.
- Priority: High

**Full dataset execution closure is not a default test:**
- What's not tested: End-to-end validation across all downloaded benchmark problems, all workloads, derived evidence sidecars, and score reports.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/readiness.py`, `src/sol_execbench/core/dataset/inventory.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`
- Risk: Small unit tests can pass while full-suite denominators, skipped-existing traces, or evidence references break.
- Priority: High

**Static evidence parser/extractor behavior relies heavily on fixtures:**
- What's not tested: Real large HIP/ELF artifacts across all ROCm tool versions and architecture targets.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `tests/sol_execbench/test_static_kernel_evidence.py`
- Risk: Tool output format changes can reduce evidence quality without breaking core benchmark execution.
- Priority: Medium

---

*Concerns audit: 2026-05-26*
