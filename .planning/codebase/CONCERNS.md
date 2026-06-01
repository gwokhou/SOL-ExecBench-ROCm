# Codebase Concerns

**Analysis Date:** 2026-06-01

## Tech Debt

**Dataset runner monolith:**
- Issue: `scripts/run_dataset.py` mixes dataset discovery, solution generation, CLI orchestration, trace parsing, derived scoring, timing evidence, and execution-closure reporting in one 1,745-line script.
- Files: `scripts/run_dataset.py`
- Impact: Changes to one reporting path can affect resume behavior, filtered workload handling, scoring sidecars, and closure provenance. It is hard to test one concern without importing a large command module.
- Fix approach: Extract stable modules under `src/sol_execbench/core/dataset/` for solution wrapping, CLI invocation, report writing, and closure assembly; keep `scripts/run_dataset.py` as argument parsing plus orchestration.

**Scoring derivation concentration:**
- Issue: SOLAR and AMD bound logic is implemented through very large heuristic modules rather than smaller operator-family units.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_bound_sanity.py`
- Impact: Operator-family additions require editing high-blast-radius files with many confidence/status paths. It is easy to add unsupported or inexact behavior without updating evidence and sanity coverage consistently.
- Fix approach: Move per-family graph extraction and estimate logic into dedicated modules under `src/sol_execbench/core/scoring/`, with shared status/confidence helpers kept in `src/sol_execbench/core/scoring/solar_derivation_status.py` and `src/sol_execbench/core/scoring/amd_bound_estimate_families.py`.

**Generated evaluation driver carries core runtime behavior:**
- Issue: `src/sol_execbench/driver/templates/eval_driver.py` is a generated standalone script but owns correctness rounds, timing, stdout redirection, reward-hack checks, safetensors loading, and trace emission.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/problem_packager.py`
- Impact: Runtime behavior is harder to type-check and reuse because it runs after staging as a subprocess script. Small changes can break JSONL output framing or subprocess error handling.
- Fix approach: Keep the staged script thin and move importable runtime helpers into `src/sol_execbench/core/bench/`; preserve subprocess isolation while unit-testing behavior through normal package imports.

**Reference/solution text rewriting workaround:**
- Issue: `scripts/run_dataset.py` rewrites every occurrence of `stream` to `strm` when wrapping reference or custom Python solutions.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/bench/reward_hack.py`
- Impact: The workaround avoids a conservative source-review rule but can silently mutate legitimate identifiers, string literals, comments, or benchmark-specific code semantics.
- Fix approach: Replace global text mutation with token-aware or AST-aware review exceptions for trusted reference paths, and add tests around string literals, comments, and variable names in `tests/sol_execbench/test_run_dataset_amd_score.py` or a new runner-focused test file.

## Known Bugs

**HIP/C++ example static-evidence validation does not prove benchmark correctness:**
- Symptoms: The documented RDNA 4 static-evidence validation run collected static artifacts but all 14 workloads returned `RUNTIME_ERROR` with `hidden_states must be a HIP tensor`.
- Files: `docs/internal/v1_17_static_kernel_evidence_validation.md`, `examples/hip_cpp/rmsnorm/`, `src/sol_execbench/cli/main.py`
- Trigger: Running `sol-execbench examples/hip_cpp/rmsnorm --solution examples/hip_cpp/rmsnorm/solution_hip.json --static-evidence auto`.
- Workaround: Treat the artifact as static-evidence-only proof. Do not use it as correctness, timing, score, paper-parity, or leaderboard evidence.

**Reference timing failures are silently hidden:**
- Symptoms: The evaluation driver reports a passed solution with `reference_latency_ms` left at `0.0` when reference timing raises an exception.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/config/benchmark_config.py`
- Trigger: Any workload where correctness succeeds but reference timing fails while `BenchmarkConfig.benchmark_reference` is true.
- Workaround: Inspect trace logs and run reference timing explicitly for benchmark-grade validation. A better fix is to emit an explicit diagnostic field or non-pass status for missing reference latency when reference benchmarking is requested.

## Security Considerations

**Submitted code executes in a subprocess, not a hardened sandbox:**
- Risk: User solution sources are imported and executed by the staged Python driver with access to the process environment, filesystem permissions, Python imports, and GPU device nodes.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/cli/main.py`, `scripts/run_docker.sh`
- Current mitigation: Static source review blocks many file I/O, process, dynamic loader, network, stream, cache, and precision downgrade patterns before import.
- Recommendations: Treat local runs as untrusted-code execution. Prefer Docker isolation through `scripts/run_docker.sh`; add explicit documentation that static review is not a sandbox; consider OS-level isolation for public service use.

**Regex-based reward-hack review is conservative but bypass-prone:**
- Risk: The static review scans stripped source text with regular expressions. It can produce false positives and can miss obfuscated or indirect behavior.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Current mitigation: Runtime integrity checks catch selected monkey-patching, lazy outputs, and thread injection.
- Recommendations: Keep regex checks as a fast gate, but add AST-based checks for Python sources and record structured source-review evidence in traces for blocked and flagged submissions.

**Diagnostic artifacts can include user-controlled compiler or runtime text:**
- Risk: CLI logs, profiler metadata, static extractor raw outputs, and trace logs preserve bounded stdout/stderr and messages that may include solution-controlled text.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/bench/static_kernel_evidence.py`, `src/sol_execbench/core/bench/rocm_profiler.py`, `src/sol_execbench/core/bench/utils.py`
- Current mitigation: CLI and static extractor outputs are size-bounded, and evidence refs are often relative to output directories.
- Recommendations: Keep all artifact writers bounded, avoid absolute paths in committed fixtures, and add regression tests when new sidecar fields include raw command output.

## Performance Bottlenecks

**Dataset execution is serial and subprocess-heavy:**
- Problem: Each problem run invokes the `sol-execbench` CLI as a subprocess, parses JSONL from stdout, and optionally runs additional profiler/evidence subprocesses.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/cli/main.py`, `src/sol_execbench/core/bench/rocm_profiler.py`
- Cause: Isolation and artifact boundaries are implemented at the process level.
- Improvement path: Add controlled problem-level parallelism for CPU-only report generation and keep GPU evaluation serial by default; expose a runner abstraction for tests and future scheduling.

**Correctness and timing multiply GPU work per workload:**
- Problem: The evaluation driver performs 10 correctness rounds, then solution timing, and then optional reference timing for every workload.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/config/benchmark_config.py`, `docs/CONFIGURATION.md`
- Cause: Benchmark rigor catches nondeterminism and reports speedup, but default full validation is expensive.
- Improvement path: Preserve defaults for benchmark-grade runs; add clearly labeled smoke/diagnostic modes that reduce correctness rounds and disable reference timing without being eligible for scoring claims.

**Static evidence recursively scans and hashes build trees:**
- Problem: Static evidence collection walks the build directory, copies artifacts, and hashes persisted files.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `tests/sol_execbench/test_static_kernel_evidence.py`
- Cause: Artifact discovery is intentionally bounded to the build root but still uses recursive traversal and per-file SHA-256.
- Improvement path: Keep traversal root-bound, but prefer explicit build artifact manifests from `src/sol_execbench/driver/templates/build_ext.py` when available.

**Derived graph scoring can execute and trace reference code:**
- Problem: Bound graph construction tries `torch.fx.symbolic_trace()` and `ShapeProp` after executing `definition.reference`; failures fall back to AST extraction.
- Files: `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/solar_derivation.py`
- Cause: Dynamic tracing improves evidence quality but is expensive and can fail for complex Python references.
- Improvement path: Cache per-definition/workload derivation sidecars, isolate FX tracing behind an explicit evidence mode, and make fallback reasons visible in aggregate reports.

## Fragile Areas

**Stdout file-descriptor redirection in staged driver:**
- Files: `src/sol_execbench/driver/templates/eval_driver.py`
- Why fragile: The driver redirects fd 1 to stderr before importing PyTorch and writes trace JSON to a saved real stdout. Any future print, logger, or import-time behavior can break JSON framing if it bypasses this convention.
- Safe modification: Route all trace output through `_emit()` and keep noisy imports after redirection. Add tests in `tests/sol_execbench/driver/test_eval_driver.py` for strict JSONL output when user code prints.
- Test coverage: Existing driver tests cover reward-hack cases; keep output framing tests close to the generated template.

**Clock-lock and GPU availability behavior depends on host state:**
- Files: `src/sol_execbench/core/bench/clock_lock.py`, `tests/conftest.py`, `scripts/run_docker.sh`, `docs/TESTING.md`
- Why fragile: Tests skip based on `/dev/kfd`, `/dev/dri`, ROCm headers, and detected gfx architecture. CI or sandbox runs can pass while skipping the hardware paths that matter most.
- Safe modification: Add CPU-safe tests for classification logic and run `requires_rocm`, `requires_rdna4`, `requires_cdna3`, and `timing_serial` checks on real hardware before claiming GPU behavior.
- Test coverage: Hardware and timing coverage is marker-gated and skipped by default unless selected with marker expressions.

**Execution closure provenance has many coupled references:**
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/dataset/evidence_refs.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`
- Why fragile: Closure records combine ready-subset filters, trace refs, solution refs, derived evidence refs, stale provenance checks, and retry behavior.
- Safe modification: Add tests for every new closure status or sidecar ref. Keep refs relative to output roots through `src/sol_execbench/core/dataset/evidence_refs.py`.
- Test coverage: There is broad closure coverage in `tests/sol_execbench/test_run_dataset_execution_closure.py`, but integration behavior still depends on runner subprocess outcomes.

**ROCm-only migration residue is managed through classification rules:**
- Files: `tests/sol_execbench/test_rocm_migration_residue_audit.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/timing.py`, `docs/`
- Why fragile: Valid compatibility namespace references such as `torch.cuda` coexist with forbidden CUDA/NVIDIA residue. The audit is regex-based and requires every new match to be classified.
- Safe modification: When adding ROCm compatibility text, update `tests/sol_execbench/test_rocm_migration_residue_audit.py` with a precise classification rather than weakening the pattern.
- Test coverage: The audit covers active roots but cannot prove runtime behavior for every compatibility namespace use.

## Scaling Limits

**Full validation requires specific ROCm hardware:**
- Current capacity: CPU-safe tests can run broadly; real GPU validation requires ROCm device nodes, ROCm PyTorch wheels, and target GPU architecture.
- Limit: `tests/conftest.py` skips `requires_rocm`, architecture-specific, and `timing_serial` tests when hardware or markers are absent.
- Scaling path: Maintain separate CPU CI and hardware CI lanes. Record RDNA 4 and CDNA 3 evidence separately under `docs/internal/` and avoid merging hardware claims from CPU-only runs.

**CDNA 3 / MI300X validation is readiness-only:**
- Current capacity: Readiness docs and diagnostics exist.
- Limit: No recorded full CDNA 3 or MI300X hardware-validation pass is present.
- Scaling path: Run the documented commands in `docs/internal/cdna3_validation_readiness.md` and `docs/internal/mi300x_validation_readiness.md` on real `gfx94*` hardware before updating public support claims.

**Dataset-scale runs produce many sidecars:**
- Current capacity: Per-problem traces, summaries, closure reports, AMD score reports, timing evidence, SOL bound sidecars, and SOLAR derivation sidecars are supported.
- Limit: Large dataset runs can create many JSON artifacts and repeated derived evidence calculations under output directories.
- Scaling path: Add manifest-based caching keyed by definition, workload UUID, hardware model, config, and git commit; keep cache provenance visible in `src/sol_execbench/core/dataset/run_closure.py`.

## Dependencies at Risk

**ROCm wheel and Docker target alignment:**
- Risk: `pyproject.toml` pins PyTorch ROCm 7.1 wheels and `triton-rocm`, while `docker/rocm-targets.json` declares multiple ROCm Docker targets.
- Impact: Mixed ROCm, PyTorch, Triton, and system ROCm versions can fail import, compile, timing, or dependency preflight.
- Migration plan: Keep dependency matrix checks in `src/sol_execbench/core/dependency_matrix.py` and `scripts/run_docker.sh` authoritative; update `docker/rocm-targets.json` and `pyproject.toml` together.

**Toolchain evidence tools are optional and version-sensitive:**
- Risk: `rocprofv3`, `llvm-objdump`, `readelf`, `hipcc`, `rocminfo`, and `rocm-smi` availability changes evidence completeness.
- Impact: Static and runtime evidence can become `unavailable`, `partial`, or `failed` without changing benchmark correctness.
- Migration plan: Keep diagnostic authority flags false in `src/sol_execbench/core/bench/static_kernel_evidence.py`; extend `src/sol_execbench/core/toolchain.py` for new tools rather than hardcoding probes in callers.

**Legacy compatibility namespaces remain necessary:**
- Risk: PyTorch ROCm uses `torch.cuda` APIs and compatibility headers, which look like CUDA residue.
- Impact: Over-aggressive cleanup can break ROCm behavior; under-aggressive cleanup can reintroduce NVIDIA-only paths.
- Migration plan: Use `tests/sol_execbench/test_rocm_migration_residue_audit.py` as the guardrail and document every intentional compatibility namespace.

## Missing Critical Features

**No full CDNA 3 / MI300X validation artifact:**
- Problem: The project targets CDNA 3, but current docs keep CDNA 3 and MI300X as deferred validation/readiness states.
- Blocks: Public claims that CDNA 3 hardware validation passed, MI300X benchmark-grade timing is validated, or FP8 behavior is validated on MI300X.
- Files: `docs/internal/cdna3_validation_readiness.md`, `docs/internal/mi300x_validation_readiness.md`, `tests/conftest.py`

**Static kernel evidence is diagnostic-only:**
- Problem: Static evidence can collect artifacts and extractor output but explicitly has no correctness, timing, score, paper-parity, or leaderboard authority.
- Blocks: Using static evidence alone to prove benchmark validity or performance.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `docs/internal/v1_17_static_kernel_evidence_validation.md`, `docs/static_kernel_evidence.md`

**No hardened untrusted execution boundary for service use:**
- Problem: The benchmark can run arbitrary submitted Python/HIP/Triton code locally through the staged driver and compiler toolchain.
- Blocks: Safe multi-tenant or public web-service execution without additional sandboxing.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/cli/main.py`, `scripts/run_docker.sh`

## Test Coverage Gaps

**Hardware-specific behavior is marker-gated:**
- What's not tested: ROCm GPU execution, architecture-specific RDNA 4/CDNA 3 behavior, ROCm native headers, CK, rocWMMA, and serial timing behavior in ordinary CPU-only runs.
- Files: `tests/conftest.py`, `tests/docker/dependencies/`, `tests/sol_execbench/core/bench/test_timing.py`, `docs/TESTING.md`
- Risk: CPU-safe test passes can mask regressions in real GPU timing, native extension build, or hardware claim behavior.
- Priority: High

**Reference timing failure path lacks explicit regression coverage:**
- What's not tested: A successful correctness path where reference timing fails and the trace still passes with zero reference latency.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: Score and speedup consumers may treat missing reference timing as a real zero-latency baseline.
- Priority: Medium

**Reward-hack defenses need continual adversarial expansion:**
- What's not tested: All possible obfuscations of process, file, import, native loader, stream, and cache abuse.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: A clever submitted solution can evade regex-based static review or runtime checks.
- Priority: High

**Large scoring heuristics require family-specific golden fixtures:**
- What's not tested: Every operator-family branch and every confidence/status transition in SOLAR and AMD bound derivation.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`, `tests/sol_execbench/test_amd_bound_estimates.py`
- Risk: Formula or evidence regressions can shift derived scores without failing high-level report shape tests.
- Priority: High

---

*Concerns audit: 2026-06-01*
