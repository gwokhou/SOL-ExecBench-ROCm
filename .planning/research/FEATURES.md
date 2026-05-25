# Feature Landscape

**Domain:** Static Kernel Evidence for a ROCm benchmark runner
**Milestone:** v1.17 Static Kernel Evidence
**Researched:** 2026-05-25
**Overall confidence:** HIGH for repo-local scope, sidecar behavior, and routing dependencies; MEDIUM for exact per-tool output shape until representative RGA, `llvm-objdump`, `roc-objdump`, and `readelf` outputs are fixture-captured on the supported ROCm 7.x toolchain.

## Scope Decision

v1.17 should add diagnostic static compiler and binary evidence to the existing ROCm-only runner. It should not create a second correctness path, a scoring input, a required benchmark artifact, or a static proof of performance. The useful product is a stable sidecar and report layer that answers: "Did this run expose a ROCm code object or HSACO?", "Which routed static tool inspected it?", "What ISA, metadata, and resource hints were extracted?", "Why was evidence unavailable?", and "How does this relate to the trace/profile/score artifacts I already have?"

The project already has the right foundation:

- v1.16 added `sol-execbench toolchain --json`, `ToolchainEvidenceLevel.STATIC`, artifact types for `rocm_binary`, `elf_object`, `hip_compiler_output`, and `triton_artifact`, plus status vocabulary for `available`, `unavailable`, `unsupported_arch`, `unsupported_artifact`, `planned`, `candidate`, `failed`, and related lifecycle states.
- v1.14 added the sidecar pattern for optional diagnostic profiler evidence next to trace output, including provenance, unavailable/failure states, artifact paths, and nonfatal fallback.
- Existing researcher docs state that canonical trace JSONL is primary and sidecars must not change trace semantics.
- Existing claim guardrails require static evidence to be linked to solution builds and interpreted by documented classification rules before it can be claimed.

The implementation should therefore treat static evidence as an optional post-build or post-run evidence collector. It should reuse the standard evaluation staging/build path where artifacts are already exposed, preserve output-derived sidecar naming, and emit explicit unsupported/unavailable records instead of failing ordinary benchmark execution.

## User-Facing Feature Contract

Users should experience Static Kernel Evidence as an opt-in diagnostic workflow:

1. Run a normal benchmark with an explicit static-evidence option or companion command.
2. The runner builds/evaluates the solution using the same ROCm path as before.
3. If the build path exposes code objects, HSACO files, or extractable ELF offload bundles, the collector records artifact references and runs the selected routed static tool.
4. If no artifact or tool is usable, the sidecar still records the reason.
5. Canonical trace JSONL, correctness, timing, scoring, and exit-code semantics remain unchanged.

Recommended interface shape:

```bash
uv run sol-execbench tests/sol_execbench/samples/rmsnorm \
  --solution tests/sol_execbench/samples/rmsnorm/solution_triton.json \
  --static-evidence auto \
  --json \
  -o out/static/rmsnorm.trace.jsonl
```

Expected sidecars:

```text
out/static/rmsnorm.trace.jsonl.static-kernel-evidence.json
out/static/rmsnorm.trace.jsonl.static/
```

The exact CLI spelling can be refined during planning, but it should be explicit and default off. A companion command such as `sol-execbench static-evidence --trace ... --artifact ...` is useful later, but the table-stakes milestone should first support evidence capture from ordinary benchmark runs where artifacts are available.

## Operator-Facing Feature Contract

Operators need deterministic artifacts they can archive with a milestone:

- `static_kernel_evidence.v1` JSON sidecar next to the trace output or inside the staging output directory when no trace path exists.
- Toolchain routing report embedded or referenced from the sidecar so the selected/static-unavailable decision is auditable.
- Artifact manifest for captured or inspected binaries: source path, persisted path, kind, sha256, size, build phase, target architecture, and whether the artifact was copied or referenced.
- Tool invocation provenance: selected tool, command, working directory, timeout, return code, stdout/stderr tails, parser version, and source refs.
- Normalized extraction results: target architecture, kernel symbols, sections/notes, code object metadata, ISA text path, resource hints when available, and classification flags.
- Human-readable report summarizing evidence status per solution/problem/workload without adding fields to canonical trace JSONL.

## Table Stakes

Features users and maintainers should expect. Missing any of these leaves v1.17 incomplete.

| Feature | Why Expected | Complexity | Dependencies on Existing Surfaces |
|---------|--------------|------------|-----------------------------------|
| Explicit opt-in static evidence collection | Static evidence is diagnostic and may require extra tools or retained build artifacts. It must not affect default benchmark runs. | Low | CLI option pattern from `--profile rocprofv3`; claim guardrails in `docs/CLAIMS.md`. |
| Default-off behavior | The milestone explicitly defers making static evidence mandatory. | Low | Existing primary `sol-execbench` behavior and tests that protect trace/default semantics. |
| Nonfatal fallback | Missing static tools, missing artifacts, parser failures, or unsupported solution types must not fail correctness evaluation. | Medium | Profile sidecar behavior in `src/sol_execbench/cli/main.py`; routing status vocabulary. |
| `static_kernel_evidence.v1` schema | Roadmap needs a machine-readable contract for static evidence and unavailable states. | Medium | Pydantic model style in `core/data`; sidecar schema patterns from environment/profile/SOLAR artifacts. |
| Diagnostic authority flags | Sidecar must say it is not correctness, performance, timing, score, paper-parity, or leaderboard authority. | Low | `ToolchainRoutingReport` authority flags and `docs/CLAIMS.md`. |
| Artifact discovery from HIP/C++ builds | Native HIP/C++ is the most controllable first path for code-object or shared-object capture. | Medium | `ProblemPackager.compile()`, staging directory, compiled artifact path returned by the packager, HIP compiler output. |
| Artifact discovery from Triton-generated kernels where stable | Triton is a core benchmark source and researchers will expect static evidence when generated artifacts are visible. | High | Existing Triton solution categories; runtime/staging cache visibility may vary by ROCm/Triton version. Emit unsupported/unavailable when not stable. |
| Artifact discovery does not mutate build semantics | Static collection must not change compiler flags or solution behavior unless explicitly documented. | Medium | Existing build commands and solution schemas. |
| Optional retained artifact copy | Researchers need archived code objects even if staging directories are removed. | Medium | Output-derived sidecar directories; `--keep-staging` remains separate from artifact persistence. |
| SHA256 and size for every copied artifact | Artifact references need reproducibility and tamper detection. | Low | Stdlib hashing; artifact manifests from profiling/dataset reports. |
| Routed static tool selection | Tool choice must go through v1.16 capability routing rather than ad hoc `shutil.which` calls. | Medium | `build_toolchain_routing_report()`, `ToolchainRoutingRequest`, static artifact types. |
| Support `llvm-objdump` extraction | Official LLVM docs support object/file disassembly and section/symbol inspection; it is a practical baseline for code-object/ELF inspection. | Medium | Toolchain registry entry for `llvm-objdump`; `ToolchainArtifactType.ELF_OBJECT` and `ROCM_BINARY`. |
| Support `readelf` metadata fallback | ELF metadata, sections, notes, and symbols are useful even when ISA disassembly is unavailable. | Medium | Registry entry for `readelf`; generic fallback status when AMD-specific tools are unavailable. |
| Support `roc-objdump` as candidate when present | Some ROCm installs expose ROCm object tools, but packaging is distribution-dependent. | Medium | `roc-objdump` candidate registry entry and probe output. |
| Support RGA as planned/preferred compiler-facing route when usable | RGA is the GPUOpen static-analysis tool most aligned with ISA/resource reporting, but availability and input modes must be proven locally. | High | Registry entry for `rga`; GPUOpen/RGA source refs; output parser fixtures. |
| Record all considered tool decisions | Researchers need to see why a fallback was selected or why no tool was selected. | Low | `ToolchainRoutingReport.decisions`. |
| Stable unavailable states | A sidecar with no evidence must still be useful and testable. | Medium | Routing statuses plus static-specific artifact states. |
| Target architecture recording | Static evidence must identify `gfx1200`, `gfx94*`, or unknown, and distinguish requested from detected target. | Medium | Solution metadata, compile flags, environment snapshot, `rocm_agent_enumerator`/toolchain request fields. |
| Code-object and HSACO metadata extraction | Core value is recording what was compiled for AMD GPU targets. | Medium | HIP compiler docs describe AMDGPU code objects in host executables or standalone `.hsaco` files. |
| Kernel symbol inventory | Researchers need to connect static output to generated kernels, not just files. | Medium | `llvm-objdump`/`readelf` symbol output; RGA output when available. |
| ISA text artifact path | Store raw disassembly output as an artifact and summarize it in JSON. | Medium | Output-derived static directory; bounded stdout/stderr capture. |
| Minimal normalized ISA classification | Useful first classifications include instruction family counts, visible MFMA/WMMA-like matrix op hints, LDS/global memory op hints, branch/barrier hints, and unknown lines. | High | Parser over disassembly text; GPUOpen/LLVM output fixtures; must be labeled heuristic. |
| Resource hint extraction when tool exposes it | VGPR, SGPR, LDS, scratch, occupancy-like hints are valuable, but tool-specific. | High | RGA output if available; metadata/notes/objdump output where possible. Emit `not_reported` when absent. |
| Explicit heuristic labels | ISA/resource classifications are diagnostics and can be incomplete or parser-dependent. | Low | Sidecar confidence/status fields. |
| Per-artifact and aggregate status | Reports should classify evidence as collected, partial, unavailable, unsupported, failed, or skipped. | Medium | Existing status/report patterns in sidecars. |
| Output-derived sidecar naming | Users expect artifacts next to `trace.jsonl` like `.profile.json` and `.environment.json`. | Low | CLI sidecar helpers in `src/sol_execbench/cli/main.py`. |
| No canonical trace JSONL mutation | Required by milestone and researcher docs. | Low | Existing trace contract and guardrail tests. |
| No score/report eligibility mutation | Static evidence is not an AMD-native score input in v1.17. | Low | Existing scoring sidecar boundaries. |
| Documentation with interpretation examples | Researchers need to know how to read collected/partial/unavailable evidence. | Low | `docs/RESEARCHER-GUIDE.md`, `docs/COOKBOOK.md`, `docs/CLAIMS.md`, `docs/rocm_toolchain_routing.md`. |
| Guardrail tests for claim language | Static analysis is easy to overstate as performance proof. | Low | Existing release docs guardrail tests. |
| CPU-only unit tests with fixture outputs | CI should not require RGA, ROCm GPU hardware, or live generated code objects. | Medium | Existing injected runner/probe patterns in toolchain/profile tests. |
| Bounded subprocess execution | Static tools can hang or emit huge output; timeouts and output tails are required. | Medium | `DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS`, profiler subprocess wrappers. |
| Path safety and output containment | Captured artifacts and reports must stay under the selected output/staging directory. | Medium | Sidecar path patterns and staging directory management. |

## Static Evidence Status Semantics

Recommended top-level sidecar status values:

| Status | Meaning | Testable Trigger |
|--------|---------|------------------|
| `collected` | At least one artifact was discovered, one routed tool succeeded, and normalized evidence was extracted. | Fixture artifact plus successful injected tool output. |
| `partial` | Artifact discovery succeeded but only metadata or only raw output could be extracted. | `readelf` succeeds but ISA parser has no disassembly, or RGA resource fields are absent. |
| `unavailable` | A required artifact or tool was not present. | No code object/HSACO found, selected tool absent, or routing selected no static tool. |
| `unsupported_solution` | The solution/build path has no stable static artifact capture contract in v1.17. | PyTorch-only eager solution, opaque library call without exposed binary, or Triton artifact unavailable. |
| `unsupported_arch` | Requested/detected architecture is outside the routed tool entry. | Routing decision returns `unsupported_arch`. |
| `unsupported_artifact` | Tool exists but does not support the artifact type. | Routing decision returns `unsupported_artifact`. |
| `failed` | Tool invocation or parser failed after artifact discovery. | Nonzero command, timeout, malformed output, decode error. |
| `skipped` | User did not request static evidence or policy disabled it. | Default run or explicit `--static-evidence none`. |

Recommended reason codes:

| Reason Code | Applies To | Meaning |
|-------------|------------|---------|
| `static_evidence_disabled` | `skipped` | User did not opt in. |
| `no_trace_output_path` | `unavailable` | Sidecar path could not be derived and no explicit path was provided. |
| `no_stable_build_artifact` | `unavailable` | Build/evaluation completed but no inspectable binary was exposed. |
| `staging_removed_before_capture` | `failed` | Collector ran after temporary staging cleanup. |
| `tool_route_unavailable` | `unavailable` | Routing found matching tools but probes did not find usable executables. |
| `tool_route_planned` | `unavailable` | Registry entry remains planned and should not be selected. |
| `tool_command_failed` | `failed` | Static tool returned nonzero. |
| `tool_command_timeout` | `failed` | Static tool exceeded timeout. |
| `metadata_only` | `partial` | ELF metadata was extracted but ISA was not. |
| `isa_parser_no_kernel_sections` | `partial` | Disassembly output lacked recognizable kernel sections or symbols. |
| `resource_metrics_not_reported` | `partial` | Tool did not expose VGPR/SGPR/LDS/scratch data. |
| `triton_artifact_unstable` | `unsupported_solution` | Triton generated artifact location is not stable under the current runtime/toolchain. |
| `library_artifact_opaque` | `unsupported_solution` | hipBLAS/MIOpen/CK/rocWMMA path invokes external libraries without a solution-owned code object. |
| `py_runtime_only` | `unsupported_solution` | Python/PyTorch path has no benchmark-owned compiled kernel artifact. |

## Sidecar Contract

The sidecar should be a strict, versioned JSON object. Recommended fields:

| Field | Type | Purpose | Complexity |
|-------|------|---------|------------|
| `schema_version` | string | `sol_execbench.static_kernel_evidence.v1`. | Low |
| `generated_at` | string | UTC timestamp. | Low |
| `diagnostic_only` | boolean | Always `true`. | Low |
| `correctness_authority` | boolean | Always `false`. | Low |
| `performance_authority` | boolean | Always `false`. | Low |
| `timing_authority` | boolean | Always `false`. | Low |
| `score_authority` | boolean | Always `false`. | Low |
| `leaderboard_authority` | boolean | Always `false`. | Low |
| `problem` | object | Definition/problem path/name when available. | Low |
| `solution` | object | Solution name, language categories, solution path, optional solution hash. | Medium |
| `run_context` | object | Trace path, output directory, staging path retained/copied flag, command summary. | Medium |
| `requested_architecture` | string/null | User/config requested `gfx*`. | Low |
| `detected_architecture` | string/null | Architecture from toolchain/environment when available. | Medium |
| `rocm_version` | string/null | Version context when available. | Medium |
| `routing` | object | Embedded `sol_execbench.toolchain_routing.v1` report or stable reference. | Low |
| `status` | enum | Aggregate sidecar status. | Low |
| `reason_codes` | list[string] | Stable reasons for skipped/unavailable/partial/failure states. | Low |
| `artifacts` | list[object] | Captured/inspected binaries and raw tool outputs. | Medium |
| `tools` | list[object] | Invoked tools with command provenance and result. | Medium |
| `kernels` | list[object] | Kernel symbols and per-kernel extracted evidence. | High |
| `classifications` | object | Aggregate heuristic ISA/resource classifications. | High |
| `warnings` | list[string] | Human-readable guardrails and degraded-state notes. | Low |
| `source_refs` | list[string] | Docs/source URLs from routing and extractor definitions. | Low |

Recommended `artifacts[]` fields:

| Field | Purpose |
|-------|---------|
| `artifact_id` | Stable local identifier. |
| `kind` | `hsaco`, `code_object`, `host_elf`, `shared_object`, `static_archive`, `isa_text`, `metadata_text`, `tool_stdout`, `tool_stderr`, or `unknown`. |
| `source_path` | Original path, redacted or relative when needed. |
| `persisted_path` | Sidecar-relative output path when copied. |
| `sha256` | Digest for copied/read artifact. |
| `size_bytes` | Artifact size. |
| `target_architecture` | `gfx*` if known. |
| `producer` | `hip_cpp_build`, `triton_runtime`, `tool_output`, or `unknown`. |
| `inspectable` | Boolean. |
| `unavailable_reason` | Optional reason when referenced but not inspectable. |

Recommended `kernels[]` fields:

| Field | Purpose |
|-------|---------|
| `kernel_id` | Stable sidecar-local ID. |
| `symbol` | Symbol or demangled name when available. |
| `artifact_id` | Artifact where symbol/evidence came from. |
| `target_architecture` | `gfx*` if known. |
| `isa_artifact_id` | Raw ISA text artifact if produced. |
| `instruction_counts` | Heuristic counts by normalized family. |
| `resource_hints` | VGPR/SGPR/LDS/scratch/etc. with `reported`, `not_reported`, or `unknown` states. |
| `classification_status` | `classified`, `partial`, `unknown`, or `not_supported`. |
| `classification_warnings` | Parser and confidence warnings. |

## Report Expectations

The human-readable report can be Markdown or a compact Rich/console summary, but the JSON sidecar is authoritative for automation. Required report sections:

| Section | Contents | Complexity |
|---------|----------|------------|
| Evidence Summary | Aggregate status, selected tool, artifact count, kernel count, architecture, key reason codes. | Low |
| Artifact Manifest | Code objects/HSACO/ELF/tool outputs with paths and hashes. | Medium |
| Tool Routing | Selected tool, fallback decisions, unavailable tools, planned/candidate lifecycle notes. | Low |
| Kernel Inventory | Kernel symbols, target arch, ISA artifact path, resource hints where available. | High |
| Classification Summary | Instruction families and static hints with explicit heuristic wording. | High |
| Unsupported/Unavailable States | Why evidence was missing and what to do next. | Medium |
| Claim Boundary | Static evidence is diagnostic and non-authoritative for correctness, performance, scoring, paper parity, or leaderboard claims. | Low |

Dataset-level reporting should be deferred unless it is a thin aggregation of per-run sidecars. If included late in v1.17, it should count `collected`, `partial`, `unavailable`, `unsupported_solution`, and `failed` by problem without requiring full paper-scale coverage.

## Differentiators

Valuable features that improve researcher confidence but should not displace table stakes.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Tool-output fixture corpus | Makes parser behavior stable across ROCm/tool versions. | Medium | Store small sanitized outputs from `llvm-objdump`, `readelf`, `roc-objdump`, and RGA when available. |
| Sidecar-to-profile correlation hints | Helps compare static kernel names with `rocprofv3` kernel activity. | High | Use symbol/kernel-name joins only as hints; do not alter timing. |
| Sidecar-to-trace references | Researchers can locate the trace and workload that produced evidence. | Medium | Keep as paths/IDs in sidecar, not trace fields. |
| ISA family taxonomy | Useful for matrix core, scalar/vector ALU, memory, LDS, branch, barrier, and conversion instruction inspection. | High | Treat as heuristic and architecture-version-sensitive. |
| Resource regression diff | Compare two static sidecars for the same solution to spot VGPR/LDS/scratch or instruction-mix changes. | Medium | Defer unless table-stakes sidecar is complete. |
| Manual artifact analysis command | Lets researchers inspect an existing `.hsaco`/ELF without running the benchmark. | Medium | Useful as `sol-execbench static-evidence --artifact path --gpu-arch gfx1200`; can follow core implementation. |
| HTML or notebook-friendly report | Easier deep inspection of ISA and resource hints. | Medium | Defer; Markdown/JSON are enough for v1.17. |
| RGA-first resource extraction | Higher-value resource data than generic ELF tools. | High | Implement only after routing and fixture validation prove stable command/output behavior. |
| Multi-architecture artifact split | Fat binaries may embed multiple code objects; split/report per architecture. | High | Useful but can be incremental. Start by reporting detected architecture entries and raw output. |
| Package/version fingerprinting | Captures `hipcc`, LLVM, RGA, and ROCm tool versions for reproducibility. | Medium | Reuse routing probes and environment snapshot where available. |

## Anti-Features

Features to explicitly not build for v1.17.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Static evidence as correctness authority | Static inspection cannot prove numerical correctness. | Keep correctness solely in canonical benchmark evaluation traces. |
| Static evidence as timing/performance authority | ISA/resource hints do not replace measured latency or profiler evidence. | Report static hints as diagnostic context beside traces and profile sidecars. |
| Static evidence as AMD-native score input | v1.17 is not a scoring milestone and static metrics are not SOL/SOLAR bounds. | Leave AMD-native score reports dependent on existing trace/bound/baseline sidecars. |
| Static evidence as leaderboard eligibility | The project does not claim hosted leaderboard readiness. | Use claim-boundary wording from `docs/CLAIMS.md`. |
| Mandatory static evidence for every run | Tool availability and artifact capture vary by solution type and ROCm installation. | Default off; emit explicit unavailable/unsupported states when requested. |
| Full 235-problem static coverage requirement | The milestone scope says full paper-scale static artifact coverage is deferred unless explicitly included. | Support per-run evidence and optional bounded aggregation. |
| Recompiling with different optimization flags just to expose artifacts | This can change benchmark behavior and invalidate trace comparison. | Prefer artifacts from the normal build path; if special flags are ever added, label them as separate diagnostic builds. |
| Mutating canonical trace JSONL | Existing public trace schemas are stable and primary. | Write sidecars only. |
| Hiding raw tool output | Normalized parsers can be wrong. | Persist bounded raw stdout/metadata/ISA artifacts where possible. |
| Treating parser heuristics as architecture facts | ISA naming and tool output can change across ROCm/LLVM/GPU generations. | Include confidence/status/warnings and raw output refs. |
| Failing benchmark execution because RGA/objdump is missing | Optional diagnostics should not break ordinary runs. | Nonfatal sidecar status `unavailable`. |
| CUDA/NVIDIA static-analysis compatibility | This is a ROCm-only port. | Analyze AMD ROCm artifacts only. |
| Deep decompilation or source reconstruction | Decompilation is outside benchmark evidence scope and risky to overinterpret. | Keep to metadata, disassembly, symbols, and simple classifications. |
| Profiler-counter roofline replacement | Static evidence is not `rocprofiler-compute` or runtime counter analysis. | Leave roofline/counter workflows for future profiling milestones. |
| Remote services or databases | Static evidence should be local, reproducible, and archive-friendly. | Use local JSON/Markdown/artifact directories. |

## Dependencies On Existing Routing, Profile, Trace, And Score Artifacts

```text
Normal benchmark input
  -> existing solution load and staging
  -> existing HIP/Triton/PyTorch/ROCm-library build or evaluation path
  -> optional static artifact discovery before staging cleanup
  -> v1.16 toolchain routing request
  -> bounded static tool invocation
  -> raw artifacts + normalized parser output
  -> static_kernel_evidence.v1 sidecar
  -> optional human-readable report

Canonical trace JSONL
  -> referenced by static sidecar when available
  -> never receives static fields

Environment sidecar / doctor output
  -> optional source for ROCm version, detected arch, and tool context
  -> not required for static evidence

Profile sidecar
  -> optional correlation context for kernel names
  -> no timing fields changed by static evidence

AMD score / SOL / SOLAR sidecars
  -> may be listed beside static evidence in researcher reports
  -> never consume static evidence in v1.17 scoring

Toolchain routing report
  -> required dependency for selected tool and unavailable states
  -> embedded or referenced inside static sidecar
```

## MVP Recommendation

Prioritize:

1. Sidecar schema, status semantics, authority flags, and sidecar path behavior.
2. HIP/C++ artifact discovery from the existing compile/staging path.
3. Routed `llvm-objdump`/`readelf` extraction with fixture-based parsers and raw output artifacts.
4. Nonfatal unavailable/unsupported states for Triton, PyTorch-only, opaque ROCm-library, and missing-tool cases.
5. Documentation and claim guardrails.

Defer:

- RGA-first rich resource extraction until command/output fixtures are stable.
- Triton generated artifact capture unless a stable artifact path exists under the current ROCm/Triton version.
- Dataset-scale aggregation beyond a simple count of per-run sidecar statuses.
- Static/profile kernel-name correlation beyond optional hints.
- Any use of static evidence in scoring, correctness, timing, or leaderboard policy.

## Suggested Phase Structure

```text
Phase 1: Static sidecar contract and CLI policy
  -> schema, status enum, sidecar paths, default-off/nonfatal behavior

Phase 2: Artifact discovery and manifest
  -> HIP/C++ artifact capture, hashes, persisted static artifact directory, unsupported states for other sources

Phase 3: Routed extractor adapters
  -> route static tools through v1.16, run bounded commands, persist raw outputs, parse metadata/ISA basics

Phase 4: Reports, docs, and guardrails
  -> researcher/cookbook updates, claims updates, report rendering, no-trace/no-score mutation tests

Phase 5: Optional richer classifications
  -> instruction-family heuristics, resource hints, fixture expansion, maybe RGA if stable
```

Phase 5 should be treated as stretch unless phases 1-4 finish cleanly.

## Sources

### Primary repo-local sources (HIGH confidence)

- `.planning/PROJECT.md` - v1.17 goal, target features, explicit deferrals, and claim boundaries.
- `.planning/MILESTONES.md` - v1.16 delivered routing and deferred static artifact implementation.
- `.planning/milestones/v1.16-MILESTONE-AUDIT.md` - confirms routing foundation passed and static evidence remains v1.17 follow-up.
- `docs/CLAIMS.md` - current allowed/disallowed claim levels and static evidence upgrade rule.
- `docs/RESEARCHER-GUIDE.md` - canonical trace and sidecar interpretation model.
- `docs/COOKBOOK.md` - existing workflows for traces, environment sidecars, profiling sidecars, routing, and derived evidence.
- `docs/rocm_toolchain_routing.md` - evidence levels, lifecycle/status vocabulary, static tool matrix, and routing claim boundary.
- `src/sol_execbench/core/toolchain.py` - current routing schema, static artifact types, registry entries, dynamic probes, and authority flags.
- `src/sol_execbench/cli/main.py` - existing environment/profile sidecar path and nonfatal diagnostic behavior.

### External official sources (MEDIUM/HIGH confidence)

- HIP compiler documentation, current as crawled 2026-05-20: AMD documents ROCm binaries/code objects, standalone `.hsaco` files, kernel symbols, target architectures, and offloading metadata inspection. https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html
- HIP porting guide, current as crawled 2026-05-19: HIP compilation can generate per-target code objects bundled into host ELF `.hip_fatbin` sections and `.hsaco` is the code-object file form. https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html
- LLVM `llvm-objdump` documentation, current as crawled 2026-05-23: `llvm-objdump` inspects object files/final linked images and supports disassembly, symbols, and section/header style output. https://llvm.org/docs/CommandGuide/llvm-objdump.html
- LLVM AMDGPU usage documentation, current as crawled 2026-04: AMDGPU code objects contain ELF note records and code object version support changes across LLVM versions. https://llvm.org/docs/AMDGPUUsage.html
- RGA repository and manual references already captured by v1.16 routing docs: RGA is the planned GPUOpen static analysis route, but v1.17 should validate local command/output behavior with fixtures before relying on rich resource fields. https://github.com/GPUOpen-Tools/radeon_gpu_analyzer and https://gpuopen.com/manuals/rga_manual/help_manual/

---
*Research completed: 2026-05-25*
*Ready for requirements scoping: yes*
