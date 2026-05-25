# Project Research Summary

**Project:** SOL ExecBench ROCm Port
**Domain:** Static kernel/compiler evidence for ROCm benchmark runs
**Milestone:** v1.17 Static Kernel Evidence
**Researched:** 2026-05-25
**Confidence:** HIGH for sidecar-first integration and repository boundaries; MEDIUM for exact static-tool output and live HSACO/code-object discovery until validated in the ROCm container.

## Executive Summary

v1.17 should add opt-in diagnostic static kernel evidence to the ROCm benchmark runner. The product is not a new evaluator or scoring system; it is a sidecar evidence layer that captures build artifacts from ROCm solution builds, routes available static-analysis tools through the v1.16 toolchain registry, extracts raw ISA/metadata where possible, and records explicit unavailable or unsupported states when evidence cannot be collected.

The recommended implementation is a narrow sidecar pipeline attached to the existing HIP/C++ compile path: after `ProblemPackager.compile()` creates `benchmark_kernel.so` in the staging directory and before staging cleanup, discover stable build artifacts, copy or register bounded evidence artifacts beside the trace output, run routed extractors such as `llvm-objdump` and `readelf`, and write `sol_execbench.static_kernel_evidence.v1`. Canonical trace JSONL, correctness, timing, scoring, paper-parity, and leaderboard semantics must remain unchanged.

The main risks are toolchain drift, unstable PyTorch ROCm build intermediates, temporary staging cleanup, and overclaiming static evidence. Mitigate them by keeping collection default-off and nonfatal, persisting routing decisions and raw outputs, modeling partial/unavailable states explicitly, copying durable artifacts outside temporary staging, and adding guardrails that static evidence is diagnostic only.

## Key Findings

### Stack Additions

No new Python framework, database, binary parser, or service dependency is recommended. v1.17 should use the existing Python 3.12 package, Pydantic v2 models, Click/Rich CLI patterns, stdlib `json`/`hashlib`/`subprocess`, and the v1.16 toolchain router.

**Core technologies:**
- Python stdlib: artifact walking, hashing, bounded subprocess execution, and JSON writing.
- Pydantic v2: strict `static_kernel_evidence.v1` models and round-trip tests.
- Existing Click/Rich CLI: explicit `--static-evidence none|auto` behavior and user-facing status.
- Existing `src/sol_execbench/core/toolchain.py`: static route selection, probes, statuses, source refs, and unavailable reasons.
- `llvm-objdump`: first general extractor for headers, symbols, offload info, and disassembly when available.
- `readelf`: fallback for ELF headers, sections, notes, symbols, and metadata when ISA extraction is unavailable.
- RGA and `roc-objdump`: candidate routes only; probe and record status, but do not make either mandatory.

Recommended new module: `src/sol_execbench/core/bench/static_kernel_evidence.py` for request/result models, artifact discovery, extractor adapters, raw-output registration, conservative classification, and optional report rendering.

### Table-Stakes Features

**Must have:**
- Explicit opt-in collection, default off, with `--static-evidence none|auto`.
- Nonfatal behavior for missing tools, missing artifacts, unsupported solution types, parser failures, and timeouts.
- `sol_execbench.static_kernel_evidence.v1` JSON sidecar with diagnostic-only authority flags.
- Durable sidecar paths beside trace output, for example `traces.jsonl.static-kernel-evidence.json` and `traces.jsonl.static-kernel-evidence/`.
- HIP/C++ artifact discovery from the current staging/build directory, starting with `benchmark_kernel.so` plus opportunistic `.hsaco`, `.co`, `.o`, `.out`, compiler intermediates, and raw tool outputs.
- SHA256, size, kind, source path, persisted path, producer, and inspectability for every registered artifact.
- Routed static-tool selection through v1.16, including all considered decisions and unavailable reasons.
- Bounded extractor execution with command provenance, tool path/version, timeout, return code, raw output refs, and stdout/stderr tails.
- Explicit aggregate and per-artifact statuses: collected, partial, unavailable, unsupported, failed, skipped.
- No mutation of canonical trace JSONL, scoring reports, timing authority, or benchmark exit-code semantics.
- Documentation and guardrail tests stating that static evidence is diagnostic only.

**Should have:**
- Minimal normalized metadata: detected `gfx*` targets, section/code-object presence, symbol inventory, raw ISA text path, and coarse `has_disassembly`/`has_metadata` booleans.
- Markdown or console report summarizing evidence status, artifact manifest, tool routing, kernel inventory, unavailable states, and claim boundary.
- Fixture corpus for `llvm-objdump`, `readelf`, `roc-objdump`, and RGA outputs so parser behavior is CPU-testable.
- Optional package/version fingerprints for ROCm, LLVM, RGA, and compiler tools via routing/environment evidence.

### Deferred Scope

Defer full RGA-first resource extraction, Triton cache capture, full paper-scale 235-problem static coverage, dataset-level static aggregation beyond simple sidecar counts, static/profile kernel-name correlation, deep instruction taxonomy, resource regression diffs, HTML/notebook reports, and any use of static evidence in scoring, correctness, timing, paper-parity, or leaderboard policy.

Triton and Python/PyTorch-only solutions should return clear `unsupported_solution` or `no_stable_artifact` states in v1.17 unless a stable artifact boundary is proven during implementation. ROCm-library categories may be represented as native-adjacent only when a solution-owned artifact is available; otherwise record an opaque-library unsupported state.

## Architecture And Data Flow

Static Kernel Evidence should be an optional diagnostic sidecar attached to the existing compile/evaluate lifecycle:

```text
sol-execbench ... --solution solution.json -o traces.jsonl --static-evidence auto
  -> load definition, workload, solution, and config
  -> create staging directory and ProblemPackager
  -> native HIP/C++ compile succeeds
  -> discover static artifacts in the current staging/build tree
  -> build ToolchainRoutingRequest(evidence_level=static, artifact_type=...)
  -> run bounded routed extractor commands when tools are available
  -> copy/register artifacts and raw outputs under traces.jsonl.static-kernel-evidence/
  -> write traces.jsonl.static-kernel-evidence.json
  -> execute benchmark normally
  -> write canonical trace JSONL unchanged
```

Major components:

1. `StaticKernelEvidenceRequest` - staging dir, primary artifact, solution metadata, output dir, target arch, timeout, compile provenance, and injected test dependencies.
2. `StaticKernelEvidenceResult` - strict sidecar model with status, authority flags, routing report, artifacts, extractor runs, classifications, warnings, and reason codes.
3. `discover_static_kernel_artifacts()` - current-staging-only discovery for `benchmark_kernel.so`, code objects, HSACO, object files, compiler outputs, and raw extracted artifacts.
4. `collect_static_kernel_evidence()` - route static tools, run bounded extractors, persist raw outputs, classify conservatively, and return a result for success, partial, unsupported, unavailable, failed, or skipped.
5. CLI sidecar helpers - derive output paths, invoke collection after compile success and before cleanup, write JSON/report files, and keep static failures nonfatal.
6. Toolchain routing updates - promote implemented static routes from planned to active/candidate while preserving probe-gated availability and unsupported statuses.

Boundary rules:

- The packager may expose native-solution checks and artifact roots, but it should not parse ISA or own extractor subprocess logic.
- The router selects/probes tools and records reasons; it should not parse static evidence.
- The collector owns artifact discovery, extraction, raw-output registration, and conservative parsing.
- Trace, evaluation, scoring, dataset execution, and AMD SOL/SOLAR reports must ignore static evidence in v1.17.

## Watch-Outs

1. **Tool availability is not stable across ROCm images** - every extractor must go through routing/probes, record command provenance, and degrade to `unavailable` without failing the benchmark.
2. **PyTorch ROCm builds may not expose standalone HSACO files** - always register `benchmark_kernel.so`, distinguish shared-object evidence from direct code-object/HSACO evidence, and treat missing HSACO as partial or unavailable, not fatal.
3. **Temporary staging can leave dangling evidence paths** - copy durable evidence beside the trace before cleanup; use sidecar-relative paths and store absolute paths only as provenance.
4. **Global cache scanning can mix artifacts across runs** - search only the current staging/build tree, tie artifacts to solution/build provenance, and avoid recursive scans of `~/.cache`, `/tmp`, or ROCm install trees.
5. **Static evidence can be overclaimed** - use `diagnostic_only: true` and false authority flags for correctness, performance, timing, score, paper parity, and leaderboard claims.
6. **Text parsers will be brittle if too ambitious** - preserve raw output, parse minimal stable facts first, label heuristics, and use `unknown` rather than false when a fact cannot be extracted.
7. **CPU-safe tests should not require GPUs or live tools** - use fake runners, fake `which`, fixture artifacts, and narrow `requires_rocm`/`requires_rocm_dev` tests only for live validation.
8. **One architecture is not evidence for all AMD targets** - model requested arches, compile flags, discovered code-object arches, and hardware validation separately per artifact.

## Recommended Phase Ordering

### Phase 1: Static Evidence Contract And Guardrails

**Rationale:** Consumers and tests need a stable sidecar contract before CLI behavior or reports can be reliable.
**Delivers:** Pydantic models, status/reason enums, authority flags, sidecar path rules, contract capability token, and no-trace/no-score guardrails.
**Addresses:** `static_kernel_evidence.v1`, diagnostic-only semantics, partial/unavailable states.
**Avoids:** Overclaiming, boolean-only schema, canonical trace mutation, score coupling.
**Research flag:** Skip extra research; repo patterns and source docs are strong.

### Phase 2: Build Artifact Discovery And Manifest

**Rationale:** Extraction is only meaningful if artifacts are tied to the exact solution build and survive staging cleanup.
**Delivers:** Packager helper APIs, native-solution coverage policy, current-staging discovery, artifact kinds, copied/persisted artifact tree, hashes, sizes, source/persisted paths, compile provenance.
**Addresses:** HIP/C++ artifact capture and durable evidence archive.
**Avoids:** Stale cache capture, dangling temp paths, assuming `benchmark_kernel.so` equals HSACO.
**Research flag:** Needs implementation validation against at least one live ROCm PyTorch extension build.

### Phase 3: Routed Static Extractor Adapters

**Rationale:** v1.17 depends on v1.16 routing; direct `subprocess.run(["tool", ...])` calls would recreate the toolchain brittleness v1.16 solved.
**Delivers:** Static routing requests, `llvm-objdump` and `readelf` extractors, optional `roc-objdump`/RGA route records, bounded command runner, raw output files, command provenance, metadata/disassembly presence classification.
**Addresses:** routed static tool selection, nonfatal unavailable states, raw output preservation.
**Avoids:** mandatory RGA, hanging extractors, unexplainable fallback behavior.
**Research flag:** Needs deeper phase research or live fixture validation for RGA and `roc-objdump` before enabling rich parsing.

### Phase 4: CLI Sidecar Integration And Reports

**Rationale:** Once contract, discovery, and extractors exist, the main benchmark command can expose a coherent opt-in workflow.
**Delivers:** `--static-evidence none|auto`, sidecar and evidence-directory writes beside trace output, no-output/keep-staging behavior, warnings that do not alter exit code, optional Markdown/text report.
**Addresses:** user-facing v1.17 workflow and operator archive needs.
**Avoids:** default behavior drift, trace/schema mutation, static failures breaking benchmark runs.
**Research flag:** Skip extra research; reuse environment/profile sidecar precedent.

### Phase 5: Documentation, Guardrails, And Live Validation

**Rationale:** Static evidence is easy to misread; docs and tests must land with live examples and claim boundaries.
**Delivers:** `docs/CLAIMS.md`, `docs/rocm_toolchain_routing.md`, architecture/researcher/cookbook updates, no-claim tests, CPU-safe fixture tests, one RDNA 4 live artifact validation record, CDNA 3 caveats if not hardware-validated.
**Addresses:** researcher interpretation, release closure, validation boundaries.
**Avoids:** claiming static evidence as correctness/performance/leaderboard authority or cross-architecture validation.
**Research flag:** Needs live ROCm validation; CDNA 3 remains a validation gap unless hardware is available.

### Phase 6: Optional Richer Classification

**Rationale:** Instruction-family counts and resource hints are useful but should not block a correct v1.17 evidence surface.
**Delivers:** Heuristic instruction families, resource hints when tools report them, parser fixture expansion, possible RGA resource extraction, optional sidecar diff helpers.
**Addresses:** differentiator features only.
**Avoids:** brittle parser-first implementation and static metrics being treated as scores.
**Research flag:** Requires deeper research and fixture capture; treat as stretch or follow-up.

### Phase Ordering Rationale

- Contract and guardrails come first because every later phase writes or interprets the sidecar.
- Artifact discovery must precede extraction because extractor choice depends on artifact kind and target architecture.
- Routed extractors must precede reports so reports summarize real route decisions and command outputs.
- CLI integration should wait until the collector can always return a stable result, including unavailable and unsupported states.
- Rich classifications should be last because raw outputs and minimal facts already satisfy the milestone while deeper parser semantics need live validation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Existing Python/Pydantic/Click/toolchain-router stack is sufficient; official docs support `llvm-objdump`, `readelf`, HIP code objects, and RGA as optional. |
| Features | HIGH | Repo-local milestone scope, v1.14 sidecar precedent, and v1.16 routing foundation strongly define table stakes and exclusions. |
| Architecture | HIGH | Existing compile/evaluate split and environment/profile sidecar pattern make the insertion point clear. |
| Artifact discovery | MEDIUM | `benchmark_kernel.so` is stable, but standalone HSACO/code-object intermediates from PyTorch ROCm builds must be validated live. |
| Extractor behavior | MEDIUM | Generic object-tool behavior is documented; exact RGA/`roc-objdump` packaging and output shape are distribution-dependent. |
| Pitfalls | HIGH | Risks are grounded in local code/docs/tests plus known ROCm tool packaging variability. |

**Overall confidence:** HIGH for the roadmap shape; MEDIUM for exact live extractor and artifact details.

### Gaps To Address

- Live HIP/C++ artifact shape: validate one RDNA 4 build tree and record whether code objects are embedded in `.so`, emitted as `.hsaco`, or only visible through object metadata.
- RGA command behavior: fixture-capture actual `rga --help`, version, and binary-analysis output before enabling rich resource parsing.
- `roc-objdump` availability: probe in the project ROCm container and keep it optional unless packaging is proven stable.
- Triton artifact stability: defer or explicitly mark unsupported until a stable ROCm/Triton cache-to-solution mapping exists.
- CDNA 3 claims: keep schema/routing support separate from live CDNA 3 validation unless hardware checks are run.

## Sources

### Primary (HIGH confidence)

- `.planning/PROJECT.md` - v1.17 goal, target features, explicit deferrals, and milestone boundaries.
- `.planning/research/STACK.md` - recommended stack additions, tool priorities, sidecar fields, and dependency exclusions.
- `.planning/research/FEATURES.md` - user/operator feature contract, table stakes, differentiators, anti-features, and status semantics.
- `.planning/research/ARCHITECTURE.md` - integration point, component boundaries, data flow, path rules, and phase build order.
- `.planning/research/PITFALLS.md` - critical pitfalls, phase warnings, verification guidance, and guardrail strategy.
- `src/sol_execbench/core/toolchain.py` - v1.16 routing schema, lifecycle/status vocabulary, static artifact types, and dynamic probes.
- `src/sol_execbench/driver/problem_packager.py` and `src/sol_execbench/driver/templates/build_ext.py` - HIP/C++ staging and PyTorch ROCm extension build flow.
- `src/sol_execbench/cli/main.py` - existing environment/profile sidecar pattern and CLI orchestration boundary.
- `docs/CLAIMS.md` and `docs/rocm_toolchain_routing.md` - claim boundaries and routing documentation.

### External (MEDIUM/HIGH confidence)

- AMD HIP compiler documentation - HIP compiler/code-object concepts and ROCm binary context.
- ROCm compiler reference - `hipcc`, `amdclang++`, `--offload-arch`, and offload architecture tooling.
- LLVM AMDGPU usage documentation - AMDGPU code object and metadata background.
- LLVM `llvm-objdump` command guide - object/final-image inspection, headers, symbols, disassembly, and offload visibility.
- GNU binutils `readelf` documentation - ELF headers, sections, notes, and symbols.
- GPUOpen Radeon GPU Analyzer manual/repository - RGA binary-analysis capability; local command/output validation still required.

---
*Research completed: 2026-05-25*
*Ready for roadmap: yes*
