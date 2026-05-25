# Architecture Research: v1.17 Static Kernel Evidence

**Project:** SOL ExecBench ROCm Port  
**Domain:** Diagnostic static compiler/kernel evidence for ROCm solution builds  
**Researched:** 2026-05-25  
**Overall confidence:** HIGH for repository integration boundaries and sidecar placement; MEDIUM for exact HSACO/code-object discovery behavior until tested against live PyTorch ROCm extension builds on RDNA 4 and CDNA 3.

## Executive Summary

Static Kernel Evidence should integrate as an optional diagnostic evidence layer attached to the existing solution build path. It should not become part of canonical trace JSONL, correctness, timing, scoring, paper-parity, or leaderboard authority. The current architecture already has the right pattern: `sol-execbench` runs canonical evaluation, and optional environment/profiling metadata is written as separate sidecars beside `--output` or under the staging directory when no output path exists.

The correct insertion point is after `ProblemPackager.compile()` succeeds for HIP/C++-family solutions and before evaluation runs. At that point the staging directory contains normalized inputs, copied sources, injected offload-arch flags, `build_ext.py`, and `benchmark_kernel.so`. Static collection should inspect those build products and any discoverable ROCm compiler artifacts, route the best available static tool through `src/sol_execbench/core/toolchain.py`, and write a `sol_execbench.static_kernel_evidence.v1` sidecar.

Build packaging should expose artifact discovery metadata, but it should not own static analysis. The new static evidence module should live under `src/sol_execbench/core/bench/` beside `rocm_profiler.py`, because it is diagnostic benchmark evidence rather than scoring, dataset inventory, or public trace data. The CLI should remain the orchestrator that chooses whether to collect static evidence, writes sidecars, and reports nonfatal unavailable states.

The v1.17 roadmap should build this in narrow vertical order: schema and artifact discovery first, routed extractors second, CLI sidecar integration third, reports/docs/guardrails last. Do not start with deep ISA classification. A useful first version is one that says: "static evidence requested; build artifact found; selected tool unavailable or available; command/provenance recorded; artifact paths registered; no score authority."

## Existing Architecture Fit

```text
Current HIP/C++ evaluation:
  CLI
    -> ProblemPackager(...)
    -> packager.compile()
       -> build_ext.py
       -> torch.utils.cpp_extension.load(..., build_directory=staging_dir)
       -> benchmark_kernel.so
    -> packager.execute()
       -> eval_driver.py
       -> canonical Trace JSONL on stdout
    -> optional sidecars:
       -> traces.jsonl.environment.json
       -> traces.jsonl.profile.json
```

Recommended v1.17 integration:

```text
Static evidence requested:
  packager.compile() success
    -> discover build artifacts in staging_dir
    -> build static routing request
    -> select object/static tool from toolchain registry
    -> run bounded extractor when available
    -> write static evidence sidecar and raw extracted artifacts
  packager.execute()
    -> canonical Trace JSONL unchanged
```

The ordering matters. Static collection should run after compilation because the build products do not exist earlier. It should run before evaluation because it can register build provenance even if evaluation later fails. It should be nonfatal unless a future explicit validation command is added; the normal benchmark exit code must still be derived from workload evaluation status.

## New Components

| Component | Location | Responsibility | Boundary |
| --- | --- | --- | --- |
| `StaticKernelEvidenceRequest` | `src/sol_execbench/core/bench/static_kernel_evidence.py` | Input object for static collection: staging dir, primary build artifact, solution metadata, output directory, selected GPU arch, timeout, and routing dependency injection. | No subprocess evaluation; no trace parsing. |
| `StaticKernelEvidenceResult` | `src/sol_execbench/core/bench/static_kernel_evidence.py` | Serializable sidecar model for status, provenance, route, artifacts, commands, stdout/stderr tails, classifications, and unavailable reasons. | Diagnostic-only fields only. |
| `StaticKernelArtifact` | `src/sol_execbench/core/bench/static_kernel_evidence.py` | One registered build or extracted artifact with path, kind, size, sha256, and optional source relation. | Paths should be stable and preferably relative in reports. |
| `discover_static_kernel_artifacts()` | `src/sol_execbench/core/bench/static_kernel_evidence.py` | Find `benchmark_kernel.so`, `.hsaco`, `.co`, object files, and likely PyTorch extension intermediates in staging/build directories. | Discovery only; do not assume every build exposes HSACO. |
| `collect_static_kernel_evidence()` | `src/sol_execbench/core/bench/static_kernel_evidence.py` | Build routing request, run selected static extractor when available, register raw outputs, and return a result even for skipped/unavailable/failed states. | Must not raise for normal tool unavailability. |
| Static report renderer | `src/sol_execbench/core/bench/static_kernel_evidence.py` or `src/sol_execbench/core/reporting.py` | Optional Markdown/text summary from the sidecar. | Reads sidecars; does not score. |

Keep this as one module first. Split into `static_tools/` only if extractor-specific code becomes large enough to justify it.

## Modified Components

| Component | Modification | Rationale |
| --- | --- | --- |
| `src/sol_execbench/cli/main.py` | Add `--static-evidence none|auto`, default `none`; add helper functions `_static_evidence_output_directory()`, `_static_evidence_sidecar_path()`, and `_write_static_evidence_sidecar()`. | Mirrors existing optional `--profile rocprofv3` sidecar pattern and keeps default evaluation unchanged. |
| `src/sol_execbench/driver/problem_packager.py` | Add public build artifact helpers such as `is_native_solution`, `primary_build_artifact_path`, and `discover_build_artifacts()`; avoid using private `_is_cpp` from CLI for new behavior. | Lets static evidence inspect packaged build outputs without moving static logic into the packager. |
| `src/sol_execbench/core/toolchain.py` | Promote static tools from `PLANNED` to `ACTIVE` or `CANDIDATE` where v1.17 implements them; preserve unavailable/unsupported statuses. Add artifact types only if needed, for example `CODE_OBJECT` or `SHARED_OBJECT_WITH_CODE_OBJECT`. | Static collection should reuse v1.16 routing instead of inventing a second tool selector. |
| `src/sol_execbench/core/data/contract.py` | Add `static_kernel_evidence.v1` capability and boundary claims; keep trace field requirements unchanged. Bump contract version to `1.1` because downstream consumers can now detect a new sidecar capability. | Makes the new sidecar discoverable without mutating `Trace`. |
| `src/sol_execbench/core/__init__.py` | Export static evidence models/helpers only if the project exports comparable optional evidence helpers there. | Keep import surface consistent. |
| `docs/CLAIMS.md` | Add allowed "Static kernel evidence" claim and forbidden wording. Remove "deferred to v1.17" once implemented. | Prevents static artifacts from being overclaimed. |
| `docs/ARCHITECTURE.md` and `docs/rocm_toolchain_routing.md` | Document sidecar flow and static routing status. | Keeps user-facing architecture aligned with implementation. |

## Data Flow

### CLI Evaluation Flow

```text
sol-execbench ... --solution solution.json -o traces.jsonl --static-evidence auto

1. Load Definition, Workload, Solution, BenchmarkConfig.
2. Create staging directory and ProblemPackager.
3. If native HIP/C++ family:
   a. packager.compile()
   b. run build_ext.py
   c. on compile failure: exit as today; optional static evidence is not written unless a later phase explicitly adds compile-failure evidence.
   d. on compile success: collect static evidence from staging/build artifacts.
4. Run evaluation normally.
5. Write canonical trace JSONL exactly as today.
6. Write optional environment/profile/static sidecars beside trace output.
7. Exit 0 only if all workloads passed; static unavailable/failed does not change this.
```

### Static Collection Flow

```text
StaticKernelEvidenceRequest
  staging_dir
  primary_artifact = staging_dir / "benchmark_kernel.so"
  solution languages/target_hardware/compile_options
  output_directory = traces.jsonl.static-kernel-evidence/ or staging_dir/static-kernel-evidence/
  routing request = evidence_level=static, artifact_type=rocm_binary or elf_object

collect_static_kernel_evidence()
  -> discover_static_kernel_artifacts()
  -> build_toolchain_routing_report()
  -> if selected tool available:
       run bounded extraction command
       write raw outputs under output_directory
       classify extracted metadata conservatively
     else:
       return status="unavailable" or "skipped" with route decisions
  -> StaticKernelEvidenceResult.to_dict()
```

### Report Flow

```text
static_kernel_evidence.v1 sidecar
  -> optional markdown/text report
  -> dataset or release closure can reference path later
  -> scoring ignores it
  -> canonical trace parser ignores it
```

## Stable Artifact Paths

Use output-relative paths when `--output` is provided:

```text
traces.jsonl
traces.jsonl.static-kernel-evidence.json
traces.jsonl.static-kernel-evidence/
  artifacts/
    benchmark_kernel.so
    <discovered>.hsaco
    <discovered>.o
  extracted/
    llvm-objdump-disassemble.txt
    readelf-headers.txt
    rga-analysis.txt
  report.md
```

Use staging-relative paths when no `--output` is provided:

```text
<staging_dir>/
  benchmark_kernel.so
  static-kernel-evidence/
    static-kernel-evidence.json
    artifacts/
    extracted/
    report.md
```

Sidecar path rule:

| Context | Sidecar path | Artifact directory |
| --- | --- | --- |
| `--output traces.jsonl` | `traces.jsonl.static-kernel-evidence.json` | `traces.jsonl.static-kernel-evidence/` |
| No `--output` and `--keep-staging` | `<staging>/static-kernel-evidence/static-kernel-evidence.json` | `<staging>/static-kernel-evidence/` |
| No `--output` and no keep-staging | Collection may run for console diagnostics, but durable artifacts are not guaranteed. Prefer warning that static evidence needs `--output` or `--keep-staging`. |

The sidecar should store both absolute command working directories for provenance and stable relative artifact paths for reproducibility. If copying raw artifacts is expensive or risks mutating build products, register paths in place for staging output and copy only when writing beside `--output`.

## Sidecar Contract

Schema version: `sol_execbench.static_kernel_evidence.v1`

Required top-level shape:

```json
{
  "schema_version": "sol_execbench.static_kernel_evidence.v1",
  "status": "success|skipped|unavailable|failed|unsupported",
  "diagnostic_only": true,
  "correctness_authority": false,
  "performance_authority": false,
  "score_authority": false,
  "leaderboard_authority": false,
  "canonical_trace_mutation": false,
  "generated_at": "ISO-8601",
  "working_directory": "/tmp/sol_execbench_xxx",
  "solution": {
    "name": "solution name",
    "languages": ["hip_cpp"],
    "target_hardware": ["gfx1200"],
    "compile_options": {}
  },
  "build": {
    "primary_artifact": "benchmark_kernel.so",
    "artifact_discovery_root": "/tmp/sol_execbench_xxx",
    "compile_command": ["python", "build_ext.py"]
  },
  "routing": {
    "schema_version": "sol_execbench.toolchain_routing.v1",
    "selected_tool_id": "llvm-objdump"
  },
  "commands": [
    {
      "tool_id": "llvm-objdump",
      "command": ["llvm-objdump", "--disassemble", "benchmark_kernel.so"],
      "returncode": 0,
      "stdout_path": "extracted/llvm-objdump-disassemble.txt",
      "stderr_tail": ""
    }
  ],
  "artifacts": [
    {
      "path": "artifacts/benchmark_kernel.so",
      "kind": "shared_object",
      "size_bytes": 12345,
      "sha256": "..."
    }
  ],
  "classifications": {
    "has_rocm_binary": true,
    "has_disassembly": true,
    "has_kernel_symbols": null,
    "notes": []
  },
  "skipped_reason": null,
  "failed_reason": null,
  "warnings": []
}
```

Use `null` for unknown classification facts. Do not infer "no kernel symbols" from a failed or unavailable extractor.

## CLI And API Surface

### CLI

Add one option to the root evaluator command:

```bash
uv run sol-execbench <problem_dir> \
  --solution solution.json \
  -o traces.jsonl \
  --static-evidence auto
```

Recommended values:

| Value | Meaning |
| --- | --- |
| `none` | Default. Do not collect static evidence. |
| `auto` | Attempt static evidence for native HIP/C++-family builds; write success, unavailable, unsupported, or failed sidecar; never alter benchmark scoring or pass/fail. |

Do not add a "required" mode in the first implementation. A required mode would couple diagnostic tooling availability to benchmark execution and conflict with the milestone guardrail that unavailable states are nonfatal.

### Programmatic API

Expose a small injectable API for tests and future dataset runners:

```python
request = StaticKernelEvidenceRequest(
    staging_dir=staging_dir,
    primary_artifact=Path(artifact_path),
    solution=solution,
    output_directory=output_dir,
    compile_command=cmd,
)
result = collect_static_kernel_evidence(request)
sidecar_path.write_text(json.dumps(result.to_dict(), sort_keys=True) + "\n")
```

The collector should accept injectable `runner`, `which`, `routing_builder`, and `now` arguments, matching the existing testability style in `toolchain.py` and `rocm_profiler.py`.

## Tool Routing Policy

Use the existing `ToolchainRoutingRequest`:

```text
evidence_level = static
artifact_type = rocm_binary for HIP/C++ build products that may contain AMDGPU code objects
artifact_type = elf_object for generic ELF/object metadata fallback
gpu_architecture = first concrete solution target or locally detected gfx target when available
```

Recommended extractor priority:

1. `llvm-objdump` for object/ELF disassembly and section visibility, because LLVM documentation defines it as an object-file and linked-image dumper with disassembly support.
2. `readelf` for generic ELF metadata fallback, because it can inspect ELF headers/sections even when AMD-specific disassembly is unavailable.
3. `roc-objdump` as a distribution-dependent ROCm candidate; route it explicitly but do not assume it exists.
4. RGA as a richer binary-analysis route once live packaging confirms command-line behavior for the artifacts this project can produce. RGA is valuable, but it should not block the first sidecar because its supported modes and packaging differ from standard ROCm compiler tools.

This priority favors a minimum useful sidecar over waiting for full RGA interpretation. The sidecar can record RGA as unavailable/candidate while still preserving `llvm-objdump` or `readelf` evidence.

## Component Boundaries

| Boundary | Rule |
| --- | --- |
| Canonical trace JSONL | Never add static fields to `Trace`, `Evaluation`, `Performance`, or `Correctness`. |
| Score authority | Static evidence must not feed `src/sol_execbench/core/scoring/` in v1.17. |
| Evaluation exit code | Static evidence unavailable/failed must not fail a benchmark run. Compile and evaluation failures behave as they do today. |
| Packager | May expose build artifacts and artifact discovery roots; must not classify ISA or own tool subprocess logic. |
| Toolchain routing | Selects/probes tools and records reasons; does not parse static evidence. |
| Static collector | Parses/extracts static artifacts; depends on routing; returns explicit status for every path. |
| Dataset/reporting | May reference static sidecars later; should not require them for existing dataset execution. |
| Docs/claims | Must say diagnostic static evidence, not correctness, performance, paper parity, or leaderboard evidence. |

## Native And Non-Native Coverage

Initial implementation should support native HIP/C++-family solution categories:

- `hip_cpp`
- `hipblas`
- `miopen`
- `ck`
- `rocwmma`

For `pytorch` and `triton`, return `status="unsupported"` with a clear reason unless a stable artifact is discovered through an explicit future path. Triton can generate code artifacts, but the current packaging flow does not expose a stable build artifact boundary like `benchmark_kernel.so`; treating it as supported too early would make the sidecar unreliable.

## Phase Build Order

1. **Schema And Guardrails**
   - Add `static_kernel_evidence.py` models with `to_dict()` serialization.
   - Add contract capability `static_kernel_evidence.v1` and boundary claims.
   - Add unit tests for diagnostic flags, status vocabulary, and no trace mutation.

2. **Build Artifact Discovery**
   - Add public packager helpers for native solution detection and primary artifact path.
   - Implement discovery for `benchmark_kernel.so`, `.hsaco`, `.co`, `.o`, `.out`, and build intermediates under the staging directory.
   - Add CPU-only tempdir tests with fixture files.

3. **Routing And Extractors**
   - Reuse `build_toolchain_routing_report()` for static evidence.
   - Implement bounded extraction commands for `llvm-objdump` and `readelf` first.
   - Keep `roc-objdump` and RGA represented through routing status until command behavior is validated with live artifacts.

4. **CLI Sidecar Integration**
   - Add `--static-evidence none|auto`.
   - Collect after compile success and before evaluation.
   - Write `traces.jsonl.static-kernel-evidence.json` and artifact directory.
   - Ensure unavailable/failed static collection prints a warning and does not alter evaluation exit.

5. **Reports And Documentation**
   - Add optional Markdown/text summary renderer.
   - Update `docs/ARCHITECTURE.md`, `docs/rocm_toolchain_routing.md`, `docs/CLAIMS.md`, and researcher guide references.
   - Add docs guardrail tests for forbidden authority claims.

6. **Live ROCm Validation**
   - Run at least one RDNA 4 HIP/C++ example with `--static-evidence auto`.
   - Archive sidecar and artifact tree shape.
   - If CDNA 3 hardware is unavailable, label CDNA 3 static evidence support as schema/routing support only.

## Tests To Add

| Test Area | Recommended Coverage |
| --- | --- |
| Schema | `StaticKernelEvidenceResult.to_dict()` includes diagnostic-only booleans and no authority flags. |
| Discovery | Finds primary `.so`, registers `.hsaco`/object fixtures, computes size and checksum deterministically. |
| Routing | Static collector records `unavailable` when no static tool is on PATH. |
| Extractor success | Inject fake runner and fake `which` to verify command, output file registration, stdout/stderr tails, and status. |
| CLI | `--static-evidence auto` writes sidecar for native builds and does not write static fields into trace JSONL. |
| Non-native | PyTorch/Triton solutions produce `unsupported` or `skipped` metadata without failure. |
| Guardrails | Docs and contract tests reject wording that makes static evidence score/correctness/leaderboard authority. |

## Pitfalls For Roadmap

### Treating `benchmark_kernel.so` As Equivalent To HSACO

PyTorch extension builds produce a shared object at a stable path today. Official HIP compiler documentation describes device code embedded into host objects and code objects or standalone `.hsaco` files, but this repository should not assume a standalone HSACO is always emitted. The sidecar should distinguish `shared_object`, `rocm_code_object`, `hsaco`, and `unknown_object`.

### Making RGA The First Hard Dependency

RGA is useful for binary analysis, and current GPUOpen documentation describes binary analysis mode for code-object binaries. However, the project already has a routing layer and can produce useful baseline evidence with generic object tools. Start with route-aware `llvm-objdump`/`readelf` extraction and add RGA as an enhancement once live artifacts confirm command-line behavior.

### Mutating Trace Or Score Contracts

Static evidence is attractive to correlate with performance, but v1.17 should not feed score reports or trace fields. Correlation reports can be a future derived artifact after static sidecars are stable.

### Failing Benchmarks For Missing Tools

Many ROCm environments will not have every object-inspection or GPUOpen tool installed. The sidecar must preserve `unavailable`, `unsupported`, and `failed` states as evidence, not turn them into benchmark failures.

### Overclassifying ISA

Do not build deep instruction taxonomy in the first milestone. Safe initial classifications are artifact presence, command success, disassembly presence, section/symbol text presence, selected tool, and warnings. Instruction-family classification needs separate validation.

## Roadmap Implications

Recommended phase structure:

1. **Static Evidence Contract Foundation** - creates sidecar schema, diagnostic flags, contract capability token, and guardrails.
   - Addresses: stable consumer contract.
   - Avoids: trace/schema churn.

2. **Build Artifact Discovery** - exposes packager artifact paths and registers stable artifacts.
   - Addresses: connection from solution build to static evidence.
   - Avoids: brittle CLI scanning with private packager details.

3. **Routed Static Extraction** - wires `toolchain.py` static routes into a collector with nonfatal unavailable states.
   - Addresses: current v1.16 routing layer becoming useful.
   - Avoids: hard dependency on one external static tool.

4. **CLI Sidecar And Reports** - adds `--static-evidence auto`, writes sidecar/artifact tree, and renders summary.
   - Addresses: user-facing milestone feature.
   - Avoids: changing default benchmark semantics.

5. **Docs, Guardrails, Live Validation** - updates architecture/claims and captures bounded live evidence.
   - Addresses: research-grade usability.
   - Avoids: overclaiming static evidence authority.

Phase ordering rationale:

- Contract and artifact discovery must precede CLI integration because the CLI needs stable result objects and paths.
- Routing/extractors must precede reports because reports should summarize real status vocabulary, not invented placeholders.
- Documentation should land with tests after the behavior is concrete enough to guard.

## Confidence Assessment

| Area | Confidence | Notes |
| --- | --- | --- |
| Integration point | HIGH | Existing CLI compile/evaluate split and sidecar helpers make the after-compile hook clear. |
| Component boundaries | HIGH | Existing environment/profile/toolchain patterns already separate diagnostic evidence from trace and score authority. |
| Artifact paths | HIGH | Existing `--output` sidecar naming establishes the durable path pattern. |
| Tool priority | MEDIUM | Official docs support `llvm-objdump`, readelf, HIP compiler code objects, and RGA binary analysis, but live artifact compatibility still needs validation. |
| Contract update | HIGH | Capability token plus unchanged trace fields matches current evaluator contract design. |
| HSACO discovery | MEDIUM | PyTorch ROCm extension builds may embed device code in the shared object; standalone HSACO discovery should be opportunistic. |

## Sources

- Repository project context: `.planning/PROJECT.md`
- Current architecture: `docs/ARCHITECTURE.md`
- Toolchain routing design: `docs/rocm_toolchain_routing.md`
- CLI sidecar patterns: `src/sol_execbench/cli/main.py`
- Build packaging path: `src/sol_execbench/driver/problem_packager.py`
- PyTorch ROCm extension build template: `src/sol_execbench/driver/templates/build_ext.py`
- Routing implementation: `src/sol_execbench/core/toolchain.py`
- Profiling sidecar model: `src/sol_execbench/core/bench/rocm_profiler.py`
- Evaluator contract: `src/sol_execbench/core/data/contract.py`
- Claims guardrails: `docs/CLAIMS.md`
- HIP compiler documentation: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html
- ROCm compiler reference: https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.2.3/reference/rocmcc.html
- LLVM `llvm-objdump` command guide: https://llvm.org/docs/CommandGuide/llvm-objdump.html
- Radeon GPU Analyzer manual: https://gpuopen.com/manuals/rga_manual/
- Radeon GPU Analyzer help manual: https://gpuopen.com/manuals/rga_manual/help_manual/
- GNU binutils `readelf` documentation: https://sourceware.org/binutils/docs/binutils/readelf.html
