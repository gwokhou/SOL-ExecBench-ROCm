# Domain Pitfalls

**Domain:** Static Kernel Evidence for the SOL ExecBench ROCm port  
**Milestone:** v1.17 Static Kernel Evidence  
**Researched:** 2026-05-25  
**Overall confidence:** HIGH for repo-specific integration, artifact, test, and claim-boundary risks from local docs/tests/code; MEDIUM for ROCm tool packaging drift because executable names and feature availability vary by ROCm distribution and host image.

## Scope Boundary

v1.17 should add diagnostic static kernel evidence on top of the v1.16 toolchain routing layer. The intended output is separate sidecar/report evidence such as `static_kernel_evidence.v1`, routed tool provenance, captured artifact refs, ISA/metadata extraction status, parser warnings, and unavailable states.

The milestone must not mutate canonical trace JSONL, change benchmark scoring, turn static evidence into correctness/performance authority, or make static evidence mandatory for every run. The current project already has this pattern for runtime snapshots, `rocprofv3` profiling sidecars, AMD-native score artifacts, and `toolchain.routing.v1`; Static Kernel Evidence should follow the same sidecar-first contract.

## Future Phase Labels

Use these roadmap phase labels when assigning prevention work:

| Phase | Name | Pitfall Ownership |
| --- | --- | --- |
| Phase 1 | Static Evidence Contract And Claim Boundary | Sidecar schema, authority flags, docs/contract guardrails. |
| Phase 2 | Artifact Capture And Build Integration | Capturing `.so`, code object, HSACO, build logs, compiler command provenance, temp/cache path handling. |
| Phase 3 | Routed Extractor Adapters | Tool selection, bounded execution, unavailable states, stdout/stderr capture, version recording. |
| Phase 4 | Parser And Classification Layer | ISA/metadata parsing, conservative classifiers, architecture-specific handling. |
| Phase 5 | Reports And Researcher Workflow | Human-readable reports, artifact refs, curated examples, cookbook updates. |
| Phase 6 | Verification Matrix And Guardrails | CPU-safe tests, sandbox behavior, ROCm hardware/container checks, doc no-claim tests. |

## Critical Pitfalls

### Pitfall 1: Treating Tool Availability As Stable Across ROCm Images

**What goes wrong:** The implementation assumes a fixed set of static tools is installed and named consistently, then fails or silently omits evidence when the local ROCm image has only some of RGA, `llvm-objdump`, `roc-objdump`, `readelf`, or LLVM tools under versioned paths.

**Why it happens:** v1.16 deliberately modeled static tools as `planned` or `candidate` in `src/sol_execbench/core/toolchain.py`. The current routing registry lists RGA, `llvm-objdump`, `roc-objdump`, and `readelf`, but static extraction has not yet proven which tools exist in the Docker image, host image, or CI environment. ROCm packaging also changes over time; some tools are generic LLVM/binutils tools, some are GPUOpen tools, and some may be absent from minimal runtime images.

**Consequences:** Static evidence becomes brittle and environment-specific. Researchers see unexplained missing reports, or worse, the benchmark starts failing in environments where normal execution still works.

**Warning signs:**
- Code calls `subprocess.run(["llvm-objdump", ...])` directly instead of going through routing/probe results.
- Static extraction failure exits the main benchmark command when the run otherwise produced valid traces.
- Reports say "static evidence unavailable" without tool id, command, path, version/probe output, status, and reason code.
- Tests only cover a developer workstation where every tool happens to be installed.

**Prevention strategy:**
- Promote static routes from `planned` only after adding explicit extractor adapters and tests for `available`, `unavailable`, `failed`, `unsupported_artifact`, and `unsupported_arch`.
- Probe tools with bounded commands before extraction and persist the selected routing decision inside the static evidence sidecar.
- Treat each extractor as optional diagnostic evidence. Missing tools must produce an `unavailable` sidecar, not a benchmark failure.
- Record executable path, command, timeout, return code, stdout/stderr tails, and source refs exactly as profiling evidence does today.

**Verification guidance:**
- Unit-test routing with fake `which`/runner functions for all static tools.
- Add CLI or core tests that generate a static sidecar with no tools installed and assert benchmark trace output remains unchanged.
- Add one ROCm-container check that validates the discovered static tool matrix, but keep it outside CPU-safe CI.

**Future phase:** Phase 3: Routed Extractor Adapters; Phase 6: Verification Matrix And Guardrails.

### Pitfall 2: Assuming PyTorch Extension Builds Expose Stable HSACO Files

**What goes wrong:** Static capture searches for a predictable `.hsaco` or code-object filename, but the current HIP/C++ path builds through `torch.utils.cpp_extension.load()` in `src/sol_execbench/driver/templates/build_ext.py`, uses the staging directory as `build_directory`, and finally normalizes only the extension output to `benchmark_kernel.so`.

**Why it happens:** The project currently needs a loadable Python extension, not compiler intermediate artifacts. PyTorch and HIP may create nested build products, platform-suffixed `.so` files, object files, temporary command files, or embedded code objects whose names and locations are not part of this repo's public contract.

**Consequences:** Static evidence works on one host and disappears on another. The extractor may analyze the Python extension wrapper instead of the embedded GPU code object, or it may capture stale artifacts from a previous build.

**Warning signs:**
- Static capture relies on `glob("*.hsaco")` and marks absence as an error.
- Evidence records only `benchmark_kernel.so` and claims ISA was extracted from the actual kernel without proving a code-object section existed.
- Tests assert exact intermediate filenames produced by PyTorch.
- Build directory cleanup deletes artifacts before the sidecar can reference them.

**Prevention strategy:**
- Define artifact capture tiers: `extension_shared_object`, `embedded_rocm_code_object`, `standalone_hsaco`, `build_log`, and `unavailable`.
- Always preserve the compiled `benchmark_kernel.so` ref for HIP/C++ static evidence, even when standalone HSACO discovery fails.
- Copy captured static artifacts to an evidence directory before staging cleanup, using stable sidecar-relative paths.
- Record whether the artifact is direct code object evidence or fallback ELF/shared-object evidence; do not collapse them into one "HSACO captured" flag.

**Verification guidance:**
- Add tests with a fake staging tree containing `.so`, `.o`, `.hsaco`, nested build files, and stale files; assert deterministic capture precedence.
- Add a test where no standalone HSACO exists but `benchmark_kernel.so` does; expected status should be degraded or partial, not failed.
- For live ROCm validation, inspect at least one HIP/C++ example and prove the report points to the artifact actually produced by that run.

**Future phase:** Phase 2: Artifact Capture And Build Integration.

### Pitfall 3: Losing Evidence Because Staging Directories Are Temporary

**What goes wrong:** The sidecar references files under `tempfile.mkdtemp(prefix="sol_execbench_")` or PyTorch build paths that are deleted by `ProblemPackager.__del__()` unless `--keep-staging` is used.

**Why it happens:** The current CLI intentionally creates disposable staging directories. Existing optional sidecars use paths next to the requested trace output or profiler output directory. Static capture must not depend on `--keep-staging` for evidence durability.

**Consequences:** Reports contain dead paths. Re-running or inspecting a completed benchmark cannot reproduce what was extracted, and downstream tools cannot archive evidence.

**Warning signs:**
- `static_kernel_evidence.v1` stores absolute paths under `/tmp/sol_execbench_*`.
- Evidence exists only when `--keep-staging` is set.
- CPU tests pass because they inspect paths before cleanup, while real CLI users receive dangling refs.

**Prevention strategy:**
- Store static evidence beside the trace output, for example `<trace>.static.json` and `<trace>.static/`, matching profiler sidecar conventions.
- If no output path is provided, write into a static subdirectory under staging but mark artifact refs as ephemeral unless explicitly copied.
- Copy files before `ProblemPackager` can clean staging, and store checksums/sizes for copied artifacts.
- Preserve raw absolute source paths only as provenance fields, not as the primary artifact refs.

**Verification guidance:**
- CLI test with `keep_staging=False`: after command completion, sidecar artifact refs should still exist outside staging.
- Test no-output mode separately and assert the sidecar clearly labels artifacts as ephemeral or unavailable.
- Add path-safety tests for output filenames with dots and nested directories, following profiler sidecar behavior.

**Future phase:** Phase 2: Artifact Capture And Build Integration; Phase 6: Verification Matrix And Guardrails.

### Pitfall 4: Polluting Global Caches Or Mixing Artifacts Across Tests

**What goes wrong:** Static evidence scans PyTorch, HIP, or user cache directories globally and captures artifacts from previous builds, parallel pytest workers, or another solution with the same extension name.

**Why it happens:** `build_ext.py` uses `name="benchmark_kernel"` for every HIP/C++ build and `build_directory=str(HERE)`. Tests already provide `tmp_cache_dir` to isolate `SOLEXECBENCH_CACHE_PATH`, but the main CLI uses a fresh temp staging dir, and static capture may be tempted to search broad cache locations.

**Consequences:** Evidence becomes nondeterministic. A sidecar may point to the wrong kernel, wrong architecture, or stale code object while the current trace still corresponds to a different solution.

**Warning signs:**
- Capture code searches `~/.cache`, `/tmp`, `/opt/rocm`, or PyTorch extension cache recursively.
- Sidecar lacks solution hash, staging dir, compile command, artifact mtime, size, or checksum.
- Tests pass only when run serially; xdist causes collisions or flaky artifact counts.

**Prevention strategy:**
- Search only the current staging/build directory by default.
- Require copied artifact checksum, size, source path, and capture timestamp.
- Tie static sidecar entries to `definition`, `solution`, workload count, and compiled artifact path from the same CLI invocation.
- Use the existing `tmp_cache_dir` pattern for tests that exercise cache-sensitive behavior.

**Verification guidance:**
- Create two fake staging directories with different artifacts and assert only the current one is captured.
- Run static-evidence tests under xdist-compatible constraints; avoid shared filenames outside `tmp_path`.
- Add a regression test with stale files older than the current compile start time and verify they are either ignored or marked as stale.

**Future phase:** Phase 2: Artifact Capture And Build Integration; Phase 6: Verification Matrix And Guardrails.

### Pitfall 5: Overclaiming Static Evidence As Correctness, Performance, Or Score Authority

**What goes wrong:** Documentation or reports imply that successful ISA extraction proves the kernel is correct, faster, paper-equivalent, hardware-validated, or leaderboard-ready.

**Why it happens:** Static artifacts feel authoritative, especially when they include ISA, resource metadata, or compiler diagnostics. Existing docs explicitly forbid treating toolchain routing or profiling as correctness/performance authority, and v1.17 has the same boundary.

**Consequences:** The project weakens its claim discipline and risks misleading researchers. Downstream consumers may use static evidence to rank submissions even when traces failed or were never run.

**Warning signs:**
- Sidecar fields named `validated`, `score`, `passed_static`, or `performance_class`.
- Reports include "kernel uses expected ISA, therefore correct/optimized".
- `docs/CLAIMS.md` is not updated with a new "Static kernel evidence" row.
- Static evidence is added to `Trace`, `EvaluationStatus`, or scoring fields.

**Prevention strategy:**
- Use explicit authority flags in the sidecar: `diagnostic_only: true`, `correctness_authority: false`, `performance_authority: false`, `score_authority: false`, `leaderboard_authority: false`.
- Define statuses like `captured`, `partial`, `unavailable`, `failed`, `unsupported`, and `not_requested`; avoid pass/fail language.
- Keep static evidence references out of canonical trace JSONL and AMD-native scoring.
- Add docs and no-claim tests for "not correctness", "not performance proof", "not paper parity", and "not leaderboard authority".

**Verification guidance:**
- Contract/doc tests should fail if static evidence claims appear without diagnostic-only boundaries.
- Serialization tests should assert authority flags are always false except `diagnostic_only`.
- Existing trace schema tests should not require updates to accept static evidence fields.

**Future phase:** Phase 1: Static Evidence Contract And Claim Boundary; Phase 6: Verification Matrix And Guardrails.

### Pitfall 6: Building Brittle Parsers For ISA And Metadata Text

**What goes wrong:** Parsers assume exact output formatting from RGA, `llvm-objdump`, `roc-objdump`, or `readelf` and break when a tool version changes column names, section order, target syntax, metadata keys, or warning text.

**Why it happens:** Static tools produce human-oriented text, and ROCm/LLVM outputs are not guaranteed to be stable enough for strict line-position parsing. The current project already uses conservative parser patterns for `rocprofv3` CSV evidence; static parsers need the same discipline.

**Consequences:** Evidence classification becomes noisy, version-dependent, and hard to debug. A harmless tool formatting change may flip an evidence status from partial to failed.

**Warning signs:**
- Parser relies on fixed line numbers or exact whole-line strings.
- Unknown metadata fields raise exceptions instead of being preserved.
- Parser outputs only derived classifications and discards raw stdout/stderr/artifact refs.
- Tests contain one golden output from one workstation only.

**Prevention strategy:**
- Preserve raw extractor artifacts or bounded raw output refs alongside parsed summaries.
- Parse for stable, minimal facts first: tool id/version, target gfx arch if present, sections/code-object presence, kernel symbol names if discoverable, and extraction warnings.
- Treat unknown fields as warnings and include them in `parser_warnings`.
- Keep classification conservative: absence of a parsed fact means `unknown`, not `false`.

**Verification guidance:**
- Unit-test parsers with multiple fixture variants: missing sections, reordered sections, extra warnings, empty stdout, nonzero exit, and unknown metadata fields.
- Fuzz small whitespace/order variations for critical parsers.
- Add schema tests that require `raw_artifact_refs` or `raw_output_tail` when parsed facts are incomplete.

**Future phase:** Phase 4: Parser And Classification Layer.

### Pitfall 7: Requiring GPUs For Static Evidence Tests That Should Be CPU-Safe

**What goes wrong:** The new test suite requires `/dev/kfd`, `/dev/dri`, PyTorch ROCm GPU visibility, `/opt/rocm`, or real HIP compilation for ordinary CI, causing the CPU-safe GitHub Actions workflow to fail or skip too much.

**Why it happens:** Static evidence sits near GPU build/runtime code, but many behaviors are pure path, schema, routing, parser, and report logic. Existing `tests/conftest.py` already separates `requires_rocm`, `requires_rocm_dev`, `requires_rdna4`, and `requires_cdna3` tests.

**Consequences:** CI either blocks on unavailable GPUs or loses meaningful coverage by marking broad tests as hardware-only.

**Warning signs:**
- Parser/schema/path tests are marked `requires_rocm`.
- Tests call real `hipcc`, `rocminfo`, or `torch.utils.cpp_extension.load()` when a fake file tree would cover the behavior.
- GitHub Actions ignore list has to grow substantially to keep CPU CI green.
- Test assertions depend on the host's actual static tool installation.

**Prevention strategy:**
- Keep static sidecar model, capture planning, parser, and routing tests CPU-safe with fake runners and fake artifacts.
- Reserve real compiler/tool invocation for narrowly marked `cpp`, `requires_rocm_dev`, or `requires_rocm` tests.
- Add sandbox-specific unavailable-state tests using fake missing `/dev/kfd` and missing tools.
- Do not make the main CLI require static evidence unless an explicit option requests it.

**Verification guidance:**
- Run the same CPU-safe test subset documented in `docs/TESTING.md`.
- Add a test that simulates no ROCm device nodes and no static tools; expected result is unavailable static evidence, not a crash.
- Add one optional Docker/GPU validation command to docs for live extraction, separate from CI.

**Future phase:** Phase 6: Verification Matrix And Guardrails.

### Pitfall 8: Making Cross-Architecture Claims From One GFX Target

**What goes wrong:** Static evidence captured on `gfx1200` is generalized to CDNA 3 (`gfx940`, `gfx941`, `gfx942`) or future CDNA 4 targets, or multi-arch builds are reported as if every embedded code object was inspected.

**Why it happens:** The current packager injects `--offload-arch` for explicit target hardware and can include multiple targets. The project supports RDNA 4 and CDNA 3 in schema/build/docs, but CDNA 3 full hardware validation remains deferred. Static tools may show multiple targets, one target, or a host wrapper depending on artifact and extractor.

**Consequences:** Reports overstate hardware coverage. A kernel may have static evidence for one architecture but not another, especially in multi-target builds.

**Warning signs:**
- Sidecar has one top-level `gpu_architecture` field for a multi-target artifact.
- Report says "CDNA 3 static evidence" based only on schema support or `--offload-arch=gfx942` injection.
- Parser assumes `gfx*` in build logs equals the architecture actually present in the captured code object.
- `LOCAL` target detection is used as proof of cross-architecture availability.

**Prevention strategy:**
- Model static evidence per artifact and per discovered/declared `gfx` target.
- Distinguish `requested_arches`, `compile_arch_flags`, `discovered_code_object_arches`, and `validated_hardware_arch`.
- For multi-target artifacts, report partial extraction per target and keep unknown targets explicit.
- Keep CDNA 3/CDNA 4 language behind existing hardware-validation guardrails unless live evidence exists.

**Verification guidance:**
- Unit-test solutions with `target_hardware=["gfx1200", "gfx942"]` and assert the sidecar can represent both targets independently.
- Add parser fixtures with one discovered arch, multiple discovered arches, and no discovered arch.
- Add doc guardrails that static evidence for one arch is not full hardware validation for another.

**Future phase:** Phase 1: Static Evidence Contract And Claim Boundary; Phase 4: Parser And Classification Layer; Phase 6: Verification Matrix And Guardrails.

## Moderate Pitfalls

### Pitfall 9: Running Static Extractors On The Wrong Artifact Type

**What goes wrong:** `readelf` or `llvm-objdump` is run on source files, build logs, Python modules, or the extension wrapper without recording that the artifact was not a direct ROCm binary/code object.

**Prevention strategy:** Use `ToolchainArtifactType` precisely. Require capture code to classify inputs as `ROCM_BINARY`, `ELF_OBJECT`, `HIP_COMPILER_OUTPUT`, or `TRITON_ARTIFACT`; unsupported combinations should produce `unsupported_artifact`.

**Warning signs:** Report says ISA extraction succeeded but the source artifact path ends in `.py`, `.json`, or `.log`.

**Verification guidance:** Tests should pass fake artifacts with different suffixes and assert extractor routing rejects unsupported types.

**Future phase:** Phase 2: Artifact Capture And Build Integration; Phase 3: Routed Extractor Adapters.

### Pitfall 10: Ignoring Triton And Python Solution Boundaries

**What goes wrong:** Static evidence is advertised for every solution language, but the implemented capture path only covers HIP/C++ extension builds. Python/PyTorch and Triton solutions either produce no static artifacts or have different cache/extraction mechanics.

**Prevention strategy:** Start with HIP/C++/ROCm-library extension artifacts where the current packager has a compile step. Report `not_applicable` or `not_implemented_for_language` for Python-only paths. Treat Triton static capture as a separate follow-up unless the milestone explicitly scopes it.

**Warning signs:** Static evidence option silently does nothing for Python solutions without a sidecar status.

**Verification guidance:** Add tests for HIP/C++ supported, Python not applicable, and Triton deferred statuses.

**Future phase:** Phase 1: Static Evidence Contract And Claim Boundary; Phase 2: Artifact Capture And Build Integration.

### Pitfall 11: Letting Extractor Timeouts Hang Benchmark Runs

**What goes wrong:** Static tools run without timeouts on large or malformed artifacts, delaying benchmark completion after evaluation already succeeded.

**Prevention strategy:** Reuse the v1.16 bounded probe pattern. Every extractor command should have a timeout, captured stderr/stdout tails, and `failed` or `timed_out` status that does not change trace correctness.

**Warning signs:** `subprocess.run(..., timeout=...)` is missing in extractor code.

**Verification guidance:** Fake runner raises `TimeoutExpired`; sidecar should record failure and the CLI should complete normally.

**Future phase:** Phase 3: Routed Extractor Adapters.

### Pitfall 12: Reconstructing Compiler Commands From Logs Instead Of Recording Them

**What goes wrong:** Reports infer compile flags and offload architectures by scraping verbose build output after the fact, missing injected flags or environment variables such as `PYTORCH_ROCM_ARCH`.

**Prevention strategy:** Record compile intent before running the build: solution target hardware, injected `hip_cflags`, `PYTORCH_ROCM_ARCH`, compile command, build directory, and artifact path. Use logs as supporting evidence only.

**Warning signs:** Sidecar has parsed `--offload-arch` but no copy of the normalized `solution.json` used for compile.

**Verification guidance:** Test explicit target, `LOCAL` target, and multi-target injection cases already covered in `test_problem_packager.py`, but assert static provenance records the normalized compile options.

**Future phase:** Phase 2: Artifact Capture And Build Integration.

### Pitfall 13: Sidecar Schema That Cannot Represent Partial Evidence

**What goes wrong:** The schema has a single boolean like `available` or `captured`, so it cannot distinguish no tool, no artifact, extraction failed, parse failed, unsupported language, unsupported architecture, or partial metadata only.

**Prevention strategy:** Design `static_kernel_evidence.v1` with independent sections for request, build provenance, captured artifacts, routing decisions, extractor runs, parsed metadata, classifications, warnings, and authority flags.

**Warning signs:** Consumers need to read a human string to know why static evidence is missing.

**Verification guidance:** Schema tests should cover at least `captured_full`, `captured_partial`, `artifact_unavailable`, `tool_unavailable`, `extractor_failed`, `parser_partial`, and `not_applicable`.

**Future phase:** Phase 1: Static Evidence Contract And Claim Boundary.

### Pitfall 14: Report Output That Encourages Comparing Static Metrics As Scores

**What goes wrong:** Reports surface instruction counts, SGPR/VGPR counts, LDS usage, or occupancy-like metadata as ranking columns without context, making static evidence look like benchmark scoring.

**Prevention strategy:** If metrics are exposed, label them as compiler diagnostics and put them in an "observed static metadata" section with warnings. Do not integrate them into AMD-native score reports or trace summaries.

**Warning signs:** Report sorted by static metric, or static metrics displayed next to latency/speedup as if equivalent.

**Verification guidance:** Docs and report snapshots should include "diagnostic only" near metric tables.

**Future phase:** Phase 5: Reports And Researcher Workflow; Phase 6: Verification Matrix And Guardrails.

## Minor Pitfalls

### Pitfall 15: Unstable Absolute Paths In Golden Tests

**What goes wrong:** Expected JSON fixtures include `/tmp/...`, user home directories, or full ROCm install paths, causing failures across machines.

**Prevention strategy:** Use relative artifact refs in sidecars and normalize source paths in tests. Preserve absolute paths only under provenance fields that tests compare with placeholders.

**Future phase:** Phase 6: Verification Matrix And Guardrails.

### Pitfall 16: Missing Documentation For Unavailable States

**What goes wrong:** Users see a missing static report and assume a benchmark failure.

**Prevention strategy:** Update `docs/CLAIMS.md`, `docs/rocm_toolchain_routing.md`, `docs/TESTING.md`, and the cookbook/researcher guide with examples of unavailable, partial, and diagnostic-only static evidence.

**Future phase:** Phase 5: Reports And Researcher Workflow.

### Pitfall 17: Storing Huge Raw Dumps Inline In JSON

**What goes wrong:** Sidecars become enormous because full disassembly text is embedded directly in JSON.

**Prevention strategy:** Store large raw outputs as files under the static evidence artifact directory and include refs, checksums, sizes, and bounded text tails in JSON.

**Future phase:** Phase 2: Artifact Capture And Build Integration; Phase 5: Reports And Researcher Workflow.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
| --- | --- | --- |
| Sidecar contract | Overclaiming authority or using boolean-only status | Add authority flags, explicit status enums, and partial/unavailable states. |
| Build integration | Temporary staging cleanup leaves dangling refs | Copy artifacts beside trace output before cleanup; record checksums. |
| Artifact capture | Capturing stale cache files | Search current staging/build tree only; tie evidence to compile provenance. |
| Tool routing | Direct subprocess calls bypass v1.16 routing | Route every extractor through registry/probe decisions and persist the decision. |
| Parser layer | Text parser brittleness | Preserve raw outputs, parse minimal stable facts, warn on unknowns. |
| Architecture support | One arch used as evidence for all AMD targets | Model requested/discovered/validated arch separately and per artifact. |
| Test strategy | GPU/tool-dependent tests in CPU CI | Fake runners/artifacts for unit tests; mark live checks narrowly. |
| Reporting | Static metadata treated as ranking signal | Keep reports diagnostic and separate from trace/scoring output. |

## Sources

- Local: `.planning/PROJECT.md` and `.planning/STATE.md` for v1.17 scope and deferred claims. Confidence: HIGH.
- Local: `.planning/milestones/v1.16-MILESTONE-AUDIT.md` for routing foundation and static-evidence deferral. Confidence: HIGH.
- Local: `docs/CLAIMS.md` for claim boundaries and static evidence upgrade rule. Confidence: HIGH.
- Local: `docs/TESTING.md` and `tests/conftest.py` for CPU-safe CI, ROCm markers, sandbox device-node skips, and cache isolation. Confidence: HIGH.
- Local: `docs/rocm_toolchain_routing.md` and `tests/sol_execbench/test_toolchain_routing.py` for lifecycle/status vocabulary and planned/candidate static tools. Confidence: HIGH.
- Local: `src/sol_execbench/driver/problem_packager.py`, `src/sol_execbench/driver/templates/build_ext.py`, and `src/sol_execbench/cli/main.py` for staging, PyTorch extension build, `benchmark_kernel.so`, temp directory cleanup, profiler sidecar precedent, and compile/evaluate flow. Confidence: HIGH.
- AMD HIP compiler documentation: https://rocm.docs.amd.com/projects/HIP/en/develop/understand/compilers.html. Confidence: MEDIUM for current ROCm compiler/tool context.
- Radeon GPU Analyzer manual: https://gpuopen.com/manuals/rga_manual/help_manual/. Confidence: MEDIUM for RGA command-line/static-analysis context.
- LLVM `llvm-objdump` command guide: https://llvm.org/docs/CommandGuide/llvm-objdump.html. Confidence: MEDIUM for generic LLVM object-inspection behavior.
- GNU `readelf` documentation: https://sourceware.org/binutils/docs/binutils/readelf.html. Confidence: MEDIUM for generic ELF metadata fallback behavior.
