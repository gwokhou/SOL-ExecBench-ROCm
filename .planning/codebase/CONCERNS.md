# Codebase Concerns

**Analysis Date:** 2026-06-01

## v1.22 Status Ledger

**Milestone outcome:** Phases 100-104 closed or narrowed the code-actionable
concerns targeted by v1.22. This file now distinguishes:

- **Fixed**: the specific defect or missing guardrail was corrected with tests.
- **Narrowed**: the area remains structurally complex, but v1.22 added seams,
  evidence, or focused regression coverage that reduces immediate risk.
- **Carried forward**: the concern is still actionable but outside the current
  milestone's implementation scope.
- **Externally deferred**: the concern requires hardware, paper-scale runs,
  hosted-service infrastructure, or hard sandboxing work beyond code cleanup.

**v1.22 evidence map:**

- Phase 100 narrowed the dataset runner monolith and fixed the global text
  rewriting workaround for solution wrapping through
  `src/sol_execbench/core/dataset/runner.py` and runner tests.
- Phase 101 fixed the hidden reference-timing failure path and narrowed eval
  driver framing risk through `core/bench/eval_runtime.py`, explicit reference
  timing diagnostics, and noisy-output JSONL tests.
- Phase 102 narrowed source-review bypass risk through AST-aware Python review,
  import-alias coverage, broader bypass tests, and structured blocking evidence.
- Phase 103 narrowed scoring/static-evidence risk through focused scoring
  fixtures and explicit static artifact manifest support.
- Phase 104 narrowed dependency, closure provenance, and marker overclaim risk
  through CPU-safe guardrail tests.

**Still explicitly out of scope:** full CDNA3/MI300X/CDNA4 validation,
paper-scale 235-problem parity, upstream SOLAR equivalence, leaderboard
readiness, hosted public-service operation, and complete hard sandboxing for
multi-tenant adversarial submissions.

## Tech Debt

**Dataset runner monolith:**
- Status: Narrowed in v1.22 Phase 100.
- Issue: `scripts/run_dataset.py` mixes dataset discovery, solution generation, CLI orchestration, trace parsing, derived scoring, timing evidence, and execution-closure reporting in one 1,745-line script.
- Files: `scripts/run_dataset.py`
- Impact: Changes to one reporting path can affect resume behavior, filtered workload handling, scoring sidecars, and closure provenance. It is hard to test one concern without importing a large command module.
- v1.22 evidence: `src/sol_execbench/core/dataset/runner.py` now owns solution wrapping, CLI invocation, report helpers, and dataset execution seams; `scripts/run_dataset.py` remains a compatibility adapter.
- Remaining work: Further split scheduling/resume policy only when parallel dataset execution is introduced.

**Scoring derivation concentration:**
- Status: Narrowed in v1.22 Phase 103; structural refactor carried forward.
- Issue: SOLAR and AMD bound logic is implemented through very large heuristic modules rather than smaller operator-family units.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `src/sol_execbench/core/scoring/amd_bound_sanity.py`
- Impact: Operator-family additions require editing high-blast-radius files with many confidence/status paths. It is easy to add unsupported or inexact behavior without updating evidence and sanity coverage consistently.
- v1.22 evidence: Focused golden tests now cover additional family estimates and AMD SOL v2 scored/degraded/unscored transitions; explicit static artifact manifests reduce static-evidence discovery ambiguity.
- Remaining work: Move per-family graph extraction and estimate logic into dedicated modules under `src/sol_execbench/core/scoring/` when changing operator-family implementation, not as part of this stewardship phase.

**Generated evaluation driver carries core runtime behavior:**
- Status: Narrowed in v1.22 Phase 101.
- Issue: `src/sol_execbench/driver/templates/eval_driver.py` is a generated standalone script but owns correctness rounds, timing, stdout redirection, reward-hack checks, safetensors loading, and trace emission.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/driver/problem_packager.py`
- Impact: Runtime behavior is harder to type-check and reuse because it runs after staging as a subprocess script. Small changes can break JSONL output framing or subprocess error handling.
- v1.22 evidence: Timing helpers moved into `src/sol_execbench/core/bench/eval_runtime.py`; tests now cover reference timing diagnostics and JSONL output framing under noisy user output.
- Remaining work: Continue moving pure helper logic out of the generated driver only when touched.

**Reference/solution text rewriting workaround:**
- Status: Fixed for dataset wrapping in v1.22 Phase 100; source-review false positives narrowed in Phase 102.
- Issue: `scripts/run_dataset.py` rewrites every occurrence of `stream` to `strm` when wrapping reference or custom Python solutions.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/bench/reward_hack.py`
- Impact: The workaround avoids a conservative source-review rule but can silently mutate legitimate identifiers, string literals, comments, or benchmark-specific code semantics.
- v1.22 evidence: Dataset wrapping now uses structured source helpers instead of global text mutation, and Python source review ignores blocked words in comments/strings while detecting real stream calls.

## Known Bugs

**HIP/C++ example static-evidence validation does not prove benchmark correctness:**
- Status: Accepted diagnostic-only limitation; carried forward.
- Symptoms: The documented RDNA 4 static-evidence validation run collected static artifacts but all 14 workloads returned `RUNTIME_ERROR` with `hidden_states must be a HIP tensor`.
- Files: `docs/internal/v1_17_static_kernel_evidence_validation.md`, `examples/hip_cpp/rmsnorm/`, `src/sol_execbench/cli/main.py`
- Trigger: Running `sol-execbench examples/hip_cpp/rmsnorm --solution examples/hip_cpp/rmsnorm/solution_hip.json --static-evidence auto`.
- Workaround: Treat the artifact as static-evidence-only proof. Do not use it as correctness, timing, score, paper-parity, or leaderboard evidence.

**Reference timing failures are silently hidden:**
- Status: Fixed in v1.22 Phase 101.
- Symptoms: The evaluation driver reports a passed solution with `reference_latency_ms` left at `0.0` when reference timing raises an exception.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/config/benchmark_config.py`
- Trigger: Any workload where correctness succeeds but reference timing fails while `BenchmarkConfig.benchmark_reference` is true.
- v1.22 evidence: Reference timing failures now flow through explicit helper diagnostics and tests assert the failure is visible when reference benchmarking is requested.

## Security Considerations

**Submitted code executes in a subprocess, not a hardened sandbox:**
- Status: Deferred hard-sandbox work; docs and evidence narrowed in v1.22 Phase 102.
- Risk: User solution sources are imported and executed by the staged Python driver with access to the process environment, filesystem permissions, Python imports, and GPU device nodes.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/core/bench/reward_hack.py`, `src/sol_execbench/cli/main.py`, `scripts/run_docker.sh`
- Current mitigation: Static source review blocks many file I/O, process, dynamic loader, network, stream, cache, and precision downgrade patterns before import.
- v1.22 evidence: README and architecture docs now state static review plus subprocess execution is not hardened sandboxing or multi-tenant isolation.
- Remaining work: Public-service or multi-tenant use still requires OS/container sandbox design outside this milestone.

**Regex-based reward-hack review is conservative but bypass-prone:**
- Status: Narrowed in v1.22 Phase 102.
- Risk: The static review scans stripped source text with regular expressions. It can produce false positives and can miss obfuscated or indirect behavior.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Current mitigation: Runtime integrity checks catch selected monkey-patching, lazy outputs, and thread injection; Python source review now uses AST-aware checks for imports, process/file/network calls, loaders, streams, cache patterns, precision downgrade, and import aliases.
- Remaining work: Continue adversarial expansion as new bypass families are discovered.

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
- Status: Narrowed in v1.22 Phase 103.
- Problem: Static evidence collection walks the build directory, copies artifacts, and hashes persisted files.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `tests/sol_execbench/test_static_kernel_evidence.py`
- Cause: Artifact discovery is intentionally bounded to the build root but still uses recursive traversal and per-file SHA-256.
- v1.22 evidence: `collect_static_kernel_artifacts()` can now consume an explicit artifact manifest and records manifest provenance in sidecar source references.
- Remaining work: Have native build tooling emit a manifest by default.

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
- Status: Narrowed in v1.22 Phase 104.
- Files: `scripts/run_dataset.py`, `src/sol_execbench/core/dataset/run_closure.py`, `src/sol_execbench/core/dataset/evidence_refs.py`, `tests/sol_execbench/test_run_dataset_execution_closure.py`
- Why fragile: Closure records combine ready-subset filters, trace refs, solution refs, derived evidence refs, stale provenance checks, and retry behavior.
- Safe modification: Add tests for every new closure status or sidecar ref. Keep refs relative to output roots through `src/sol_execbench/core/dataset/evidence_refs.py`.
- v1.22 evidence: Contract tests now cover sorted sidecar source refs, derived sidecar/cache provenance, checksum sensitivity, and order-insensitive requested evidence comparison.
- Remaining risk: Integration behavior still depends on runner subprocess outcomes.

**ROCm-only migration residue is managed through classification rules:**
- Files: `tests/sol_execbench/test_rocm_migration_residue_audit.py`, `src/sol_execbench/core/data/solution.py`, `src/sol_execbench/core/bench/timing.py`, `docs/`
- Why fragile: Valid compatibility namespace references such as `torch.cuda` coexist with forbidden CUDA/NVIDIA residue. The audit is regex-based and requires every new match to be classified.
- Safe modification: When adding ROCm compatibility text, update `tests/sol_execbench/test_rocm_migration_residue_audit.py` with a precise classification rather than weakening the pattern.
- Test coverage: The audit covers active roots but cannot prove runtime behavior for every compatibility namespace use.

## Scaling Limits

**Full validation requires specific ROCm hardware:**
- Status: Externally deferred hardware validation; marker guardrails narrowed in v1.22 Phase 104.
- Current capacity: CPU-safe tests can run broadly; real GPU validation requires ROCm device nodes, ROCm PyTorch wheels, and target GPU architecture.
- Limit: `tests/conftest.py` skips `requires_rocm`, architecture-specific, and `timing_serial` tests when hardware or markers are absent.
- Scaling path: Maintain separate CPU CI and hardware CI lanes. Record RDNA 4 and CDNA 3 evidence separately under `docs/internal/` and avoid merging hardware claims from CPU-only runs.

**CDNA 3 / MI300X validation is readiness-only:**
- Status: Externally deferred.
- Current capacity: Readiness docs and diagnostics exist.
- Limit: No recorded full CDNA 3 or MI300X hardware-validation pass is present.
- Scaling path: Run the documented commands in `docs/internal/cdna3_validation_readiness.md` and `docs/internal/mi300x_validation_readiness.md` on real `gfx94*` hardware before updating public support claims.

**Dataset-scale runs produce many sidecars:**
- Status: Carried forward; cache provenance narrowed in v1.22 Phase 104.
- Current capacity: Per-problem traces, summaries, closure reports, AMD score reports, timing evidence, SOL bound sidecars, and SOLAR derivation sidecars are supported.
- Limit: Large dataset runs can create many JSON artifacts and repeated derived evidence calculations under output directories.
- v1.22 evidence: Closure contract tests now assert derived cache provenance is retained and checksum-sensitive.
- Remaining work: Add manifest-based caching keyed by definition, workload UUID, hardware model, config, and git commit.

## Dependencies at Risk

**ROCm wheel and Docker target alignment:**
- Status: Narrowed in v1.22 Phase 104.
- Risk: `pyproject.toml` pins PyTorch ROCm 7.1 wheels and `triton-rocm`, while `docker/rocm-targets.json` declares multiple ROCm Docker targets.
- Impact: Mixed ROCm, PyTorch, Triton, and system ROCm versions can fail import, compile, timing, or dependency preflight.
- v1.22 evidence: Dependency policy tests now tie default Docker target policy values to `pyproject.toml` and `uv.lock`, and assert declared policy IDs and ROCm index metadata are aligned.
- Remaining work: Update `docker/rocm-targets.json`, `pyproject.toml`, and lockfiles together for future ROCm target changes.

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
- Status: Externally deferred; not fixed by v1.22.
- Problem: The project targets CDNA 3, but current docs keep CDNA 3 and MI300X as deferred validation/readiness states.
- Blocks: Public claims that CDNA 3 hardware validation passed, MI300X benchmark-grade timing is validated, or FP8 behavior is validated on MI300X.
- Files: `docs/internal/cdna3_validation_readiness.md`, `docs/internal/mi300x_validation_readiness.md`, `tests/conftest.py`

**No CDNA4 validation artifact:**
- Status: Externally deferred; not fixed by v1.22.
- Problem: CDNA4 validation remains a future hardware-evidence task and has no
  pytest marker shortcut or recorded validation artifact in this repository.
- Blocks: Public claims that CDNA4 execution, timing, or scoring behavior has
  been validated.
- Files: `.planning/REQUIREMENTS.md`, `tests/conftest.py`, `docs/`

**Static kernel evidence is diagnostic-only:**
- Status: Accepted diagnostic-only boundary; manifest support narrowed discovery risk in v1.22 Phase 103.
- Problem: Static evidence can collect artifacts and extractor output but explicitly has no correctness, timing, score, paper-parity, or leaderboard authority.
- Blocks: Using static evidence alone to prove benchmark validity or performance.
- Files: `src/sol_execbench/core/bench/static_kernel_evidence.py`, `docs/internal/v1_17_static_kernel_evidence_validation.md`, `docs/static_kernel_evidence.md`

**No hardened untrusted execution boundary for service use:**
- Status: Externally deferred; docs clarified in v1.22 Phase 102.
- Problem: The benchmark can run arbitrary submitted Python/HIP/Triton code locally through the staged driver and compiler toolchain.
- Blocks: Safe multi-tenant or public web-service execution without additional sandboxing.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `src/sol_execbench/cli/main.py`, `scripts/run_docker.sh`

## Test Coverage Gaps

**Hardware-specific behavior is marker-gated:**
- Status: Accepted hardware reality; guardrails narrowed overclaim risk in v1.22 Phase 104.
- What's not tested: ROCm GPU execution, architecture-specific RDNA 4/CDNA 3 behavior, ROCm native headers, CK, rocWMMA, and serial timing behavior in ordinary CPU-only runs.
- Files: `tests/conftest.py`, `tests/docker/dependencies/`, `tests/sol_execbench/core/bench/test_timing.py`, `docs/TESTING.md`
- Risk: CPU-safe test passes can mask regressions in real GPU timing, native extension build, or hardware claim behavior.
- v1.22 evidence: Marker audit tests now assert timing tests require explicit `-m timing_serial` and that MI300X/CDNA4 validation shortcuts are not present.
- Priority: High

**Reference timing failure path lacks explicit regression coverage:**
- Status: Fixed in v1.22 Phase 101.
- What's not tested: A successful correctness path where reference timing fails and the trace still passes with zero reference latency.
- Files: `src/sol_execbench/driver/templates/eval_driver.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: Score and speedup consumers may treat missing reference timing as a real zero-latency baseline.
- v1.22 evidence: Eval driver tests now cover explicit reference timing failure reporting.
- Priority: Closed

**Reward-hack defenses need continual adversarial expansion:**
- Status: Narrowed in v1.22 Phase 102; continual expansion carried forward.
- What's not tested: All possible obfuscations of process, file, import, native loader, stream, and cache abuse.
- Files: `src/sol_execbench/core/bench/reward_hack.py`, `tests/sol_execbench/core/bench/test_reward_hack.py`, `tests/sol_execbench/driver/test_eval_driver.py`
- Risk: A clever submitted solution can evade regex-based static review or runtime checks.
- Priority: High

**Large scoring heuristics require family-specific golden fixtures:**
- Status: Narrowed in v1.22 Phase 103.
- What's not tested: Every operator-family branch and every confidence/status transition in SOLAR and AMD bound derivation.
- Files: `src/sol_execbench/core/scoring/solar_derivation.py`, `src/sol_execbench/core/scoring/amd_bound_graph.py`, `src/sol_execbench/core/scoring/amd_bound_estimates.py`, `tests/sol_execbench/test_solar_derivation_evidence.py`, `tests/sol_execbench/test_amd_bound_estimates.py`
- Risk: Formula or evidence regressions can shift derived scores without failing high-level report shape tests.
- v1.22 evidence: Focused tests now cover an additional family golden fixture, AMD SOL v2 scored/degraded/unscored transitions, and static artifact manifest behavior.
- Priority: Medium

---

*Concerns audit: 2026-06-01*
