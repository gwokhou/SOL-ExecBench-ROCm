# Technology Stack

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.17 Static Kernel Evidence
**Researched:** 2026-05-25
**Scope:** Stack additions and integration points for capturing ROCm code
objects / HSACO where available, extracting static ISA and metadata through
routed tools, and publishing diagnostic sidecars without changing canonical
trace JSONL, correctness, timing, scoring, paper-parity, or leaderboard
semantics.
**Overall confidence:** HIGH for repository integration points and sidecar
architecture; HIGH for `llvm-objdump`, `readelf`, HIP compiler, and AMDGPU code
object facts from official docs; MEDIUM for `roc-objdump` packaging because it
is distribution-dependent and not well covered by current public ROCm docs.

## Recommendation

Do not add a new Python framework or database dependency for v1.17. Add a
small internal static-evidence module that reuses the v1.16 toolchain router,
the existing HIP/C++ staging directory, and the established sidecar/report
pattern.

The stack shape should be:

1. `ProblemPackager.compile()` produces `benchmark_kernel.so` in the staging
   directory for HIP/C++ solutions.
2. A new artifact collector scans that staging directory after successful
   compile for stable generated files: `benchmark_kernel.so`, `*.hsaco`,
   `*.o`, `*.out`, `*.hipfb`, `*.bc`, and relevant saved compiler
   intermediates when present.
3. A routed extractor selects static tools using
   `build_toolchain_routing_report()` with `evidence_level=static` and artifact
   types such as `rocm_binary`, `elf_object`, and `hip_compiler_output`.
4. Extractors run bounded subprocess commands, store raw outputs under a
   stable evidence directory, and summarize status, provenance, tool versions,
   artifact hashes, section/symbol/ISA availability, and classification results
   in `static_kernel_evidence.v1`.
5. The CLI writes a sidecar next to the trace output, analogous to existing
   environment and `rocprofv3` sidecars.

Required output paths should be deterministic:

| Artifact | Recommended path |
|----------|------------------|
| Static sidecar next to trace | `<trace>.static.json` |
| Static evidence directory next to trace | `<trace>.static/` |
| Static sidecar without `--output` | `<staging>/static/static.json` |
| Raw extractor outputs | `<trace>.static/<artifact-stem>/<tool-id>.<kind>.txt` |
| Captured compiler artifacts | `<trace>.static/artifacts/<original-name>` or relative references into kept staging |

Prefer copying only small, relevant compiler artifacts into the evidence
directory. For large or unstable staging trees, record paths, sizes, hashes, and
classification instead of mirroring the full build directory.

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | `>=3.12,<3.14` | Static artifact discovery, bounded subprocess extraction, hashing, report generation. | Existing package baseline; stdlib is enough for file walking, subprocesses, JSON, and hashing. |
| Pydantic | `>=2.12.5` | `StaticKernelEvidence`, `StaticArtifactRef`, `StaticExtractionResult`, and classification models. | Existing public and sidecar contracts use Pydantic v2. The sidecar should be strict and round-trippable. |
| Click / Rich | Existing dependency | Add opt-in CLI flag and status messages. | The primary CLI already uses Click/Rich; no new CLI stack needed. |
| stdlib `json` | Python bundled | Write sidecar and report payloads. | Keeps evidence files diffable and dependency-free. |
| stdlib `hashlib` | Python bundled | SHA256 artifact provenance. | Static evidence must identify exactly which binary/object was inspected. |
| stdlib `subprocess` | Python bundled | Bounded external tool execution. | Matches the v1.16 router and v1.14 profiler pattern. |

### ROCm / Static Tools

| Tool | Required for v1.17? | Artifact types | Recommendation |
|------|---------------------|----------------|----------------|
| `llvm-objdump` | Required route, optional local availability | `rocm_binary`, `elf_object` | Promote from planned to active/candidate for static extraction. Use for headers, symbols, disassembly, and offload bundle inspection where supported. |
| `readelf` | Required fallback route, optional local availability | `rocm_binary`, `elf_object` | Promote from planned to active/candidate fallback for ELF headers, sections, notes, symbols, and architecture-specific metadata. |
| `rga` | Required route, optional local availability | Code object / HSACO, compiler output | Promote from planned to candidate. Use only when a precompiled AMD GPU code object is available and the local RGA binary supports the required mode. |
| `roc-objdump` | Optional route | `rocm_binary`, `elf_object` | Keep as candidate and distribution-dependent. Probe and use if present, but do not make it required. |
| `hipcc` / `amdclang++` | Existing indirect build tools | HIP compiler output | Do not call directly from the benchmark path for v1.17. Static capture should observe artifacts produced by PyTorch extension builds; optional saved-temporary flags may be passed through existing `hip_cflags`. |
| `offload-arch` / `amdgpu-offload-arch` | Optional future helper | Executable/offload image requirements | Defer unless needed for richer compatibility diagnostics. The v1.17 minimum can use existing gfx detection and object metadata. |

### Repository Integration Points

| File / Module | Stack change |
|---------------|--------------|
| `src/sol_execbench/core/toolchain.py` | Change static entries from `planned` to active/candidate where v1.17 implements extractors. Keep lifecycle/status explicit: `available`, `unavailable`, `failed`, `unsupported_artifact`, `unsupported_arch`. Add source refs for official docs. |
| New `src/sol_execbench/core/static_evidence.py` | Add Pydantic models, artifact discovery, extractor command builders, bounded command runner, raw-output registration, and classification summarization. |
| `src/sol_execbench/driver/problem_packager.py` | Keep compile contract returning `benchmark_kernel.so`. Add no broad rewrite. If needed, expose a small helper that returns candidate static artifact paths from `output_dir` after compile. |
| `src/sol_execbench/driver/templates/build_ext.py` | Avoid direct changes unless artifact capture proves impossible. Prefer passing optional compiler flags through existing `compile_options.hip_cflags`. |
| `src/sol_execbench/cli/main.py` | Add an explicit opt-in such as `--static-evidence` or `--static-evidence auto`; write sidecars after successful HIP/C++ compile and before staging cleanup. Static failure must be nonfatal. |
| `src/sol_execbench/core/data/contract.py` | Advertise optional capability `static_kernel_evidence.v1`; do not bump evaluator contract version unless the public contract requires consumers to understand it. |
| `docs/rocm_toolchain_routing.md` | Update static tools from deferred to implemented diagnostic routes and document fallback order. |
| `docs/CLAIMS.md` / researcher docs | Add claim boundary: static evidence is compiler/static diagnostic evidence only. |

## Artifact Capture Strategy

### HIP/C++ compiled extensions

The existing stack compiles HIP/C++ solutions through
`torch.utils.cpp_extension.load(..., build_directory=str(HERE), verbose=True)`
and then renames the extension to `benchmark_kernel.so`. This gives v1.17 a
controlled staging directory. Use it as the capture root.

Required collector behavior:

| Step | Behavior |
|------|----------|
| Identify root | Use `ProblemPackager.output_dir` / CLI `staging_dir` after successful compile. |
| Find primary binary | Always register `benchmark_kernel.so` if present. |
| Find code objects | Glob for common ROCm outputs, especially `*.hsaco`, `*.o`, and generated files containing `gfx*`, `hipv`, or `amdgcn` in the name. |
| Verify files | Record relative path, size, SHA256, suffix, first matching artifact type, and whether copied into evidence dir. |
| Avoid overcapture | Do not copy source files, `solution.json`, workload data, Python cache, or the full build tree by default. |
| Capture saved temps | Support users adding `--save-temps` or `-save-temps=cwd` through existing `compile_options.hip_cflags`; record whether saved temps were observed. Do not force these flags globally until compile stability is tested. |

Static evidence should be best-effort. If only `benchmark_kernel.so` exists,
emit an `available_artifact:no_code_object_found` or equivalent classification
and still run generic ELF inspection against the `.so` if useful.

### Python / Triton solutions

Do not promise complete static evidence for Python/Triton solutions in v1.17.
Triton has its own cache/artifact lifecycle and should be a future extension.
For v1.17, emit `unsupported_solution_type` or `no_stable_artifact` unless a
phase explicitly scopes Triton cache capture.

## Extractor Stack

### `llvm-objdump`

Use `llvm-objdump` as the primary general extractor when available:

| Output | Candidate command |
|--------|-------------------|
| File/section/symbol headers | `llvm-objdump --all-headers <artifact>` |
| Disassembly | `llvm-objdump --disassemble <artifact>` |
| HIP/offload bundle details | `llvm-objdump --offloading <artifact>` when accepted by local version |
| Version probe | `llvm-objdump --version` |

Bound stdout/stderr tails in the sidecar and save full raw output to files under
the static evidence directory. If disassembly exits nonzero but headers work,
classify partial evidence rather than failing the benchmark.

### `readelf`

Use `readelf` as the metadata fallback:

| Output | Candidate command |
|--------|-------------------|
| Header / program / section summary | `readelf --headers --wide <artifact>` |
| Notes | `readelf --notes --wide <artifact>` |
| Symbols | `readelf --symbols --wide <artifact>` |
| Version probe | `readelf --version` |

`readelf` should not be treated as an ISA disassembler. Its role is ELF
metadata: code object ABI hints, machine/flags, sections, symbols, and notes.

### RGA

Use RGA as a candidate high-value route for precompiled AMD GPU code objects,
not as a mandatory dependency. Official RGA docs describe binary analysis mode
for loading AMD GPU Code Object binaries and producing disassembly/resource
views. The repository should therefore probe RGA and record explicit
unavailable/unsupported states instead of assuming it is installed in ROCm
containers.

Do not hard-code a single RGA CLI command from memory. Build the adapter around
local `rga --help` / `rga --version` probes and tests with fixture runners, then
document the exact supported command once validated in the project container.
The sidecar should include the actual command used.

### `roc-objdump`

Keep `roc-objdump` as optional. If present, it can be tried before or after
`llvm-objdump` depending on local behavior, but v1.17 should not depend on it
for minimum evidence because current public ROCm documentation is sparse and
packaging varies.

## `static_kernel_evidence.v1` Sidecar

Recommended top-level fields:

| Field | Purpose |
|-------|---------|
| `schema_version` | `sol_execbench.static_kernel_evidence.v1` |
| `generated_at` | UTC timestamp |
| `diagnostic_only` | Always `true` |
| `correctness_authority` | Always `false` |
| `performance_authority` | Always `false` |
| `score_authority` | Always `false` |
| `leaderboard_authority` | Always `false` |
| `canonical_trace_jsonl` | Always `false` |
| `solution_name` / `languages` | Link evidence to solution class without changing trace schema |
| `staging_dir` | Original compile staging directory, nullable if removed |
| `evidence_dir` | Directory containing raw extraction outputs |
| `routing_report` | Embedded or referenced v1.16 static routing report |
| `artifacts` | Captured/inspected artifact refs with hash, size, type, relative path |
| `extractions` | One result per tool/artifact/output kind |
| `classification` | Summary status: `complete`, `partial`, `unavailable`, `unsupported`, `failed` |
| `warnings` | Stable warning strings for missing code object, unsupported language, tool failure, etc. |

Recommended extraction result fields:

| Field | Purpose |
|-------|---------|
| `tool_id` / `tool_path` / `tool_version` | Provenance |
| `artifact_ref` | Which file was inspected |
| `command` | Exact bounded command |
| `status` | `success`, `partial`, `unavailable`, `unsupported`, `failed`, `timeout` |
| `returncode` | Process return code |
| `timeout_seconds` | Bound used |
| `stdout_tail` / `stderr_tail` | Bounded inline diagnostics |
| `raw_output_path` | Full saved output |
| `detected_kernel_symbols` | Best-effort symbol names |
| `detected_gfx_arches` | Best-effort `gfx*` matches from metadata/output |
| `has_isa` / `has_metadata` | Coarse evidence booleans |

## Required Stack Changes

| Area | Required change |
|------|-----------------|
| Tool routing | Activate static routes implemented in v1.17. Keep probe-gated selection; no static tool is assumed installed. |
| Static models | Add sidecar Pydantic models and JSON serializer tests. |
| CLI | Add explicit opt-in static evidence collection and sidecar path helpers. |
| Artifact discovery | Add deterministic post-compile collector for HIP/C++ staging outputs. |
| Extractors | Add bounded subprocess adapters for `llvm-objdump`, `readelf`, and optional RGA / `roc-objdump`. |
| Reporting | Add raw output registration and sidecar summary; optionally add a small Markdown report generated from the sidecar. |
| Contract metadata | Add optional capability only. |
| Docs and guardrails | Update claims, routing docs, researcher docs, and tests that prevent static evidence from being described as score/correctness/performance authority. |

## Optional / Future Tools

| Tool / Capability | Recommendation |
|-------------------|----------------|
| `clang-offload-bundler` | Future helper for extracting embedded HIP offload bundles if `llvm-objdump --offloading` and build-tree capture are insufficient. Do not add in the minimum v1.17 stack. |
| `llvm-readobj` / `llvm-readelf` | Future structured ELF alternative if `readelf` output parsing becomes fragile. Not needed for first sidecar. |
| `offload-arch -f <binary>` | Future compatibility diagnostics for compiled offload images. Useful, but not required for sidecar MVP. |
| Triton cache capture | Future milestone. It has separate cache invalidation and source-to-artifact mapping concerns. |
| RGA resource/stat parsing | Future enhancement after raw RGA command behavior is validated in CI/container. v1.17 should capture raw outputs and coarse classifications first. |
| YAML / MessagePack metadata parser | Future richer parser. For v1.17, prefer raw note/section output and coarse detection; do not add PyYAML/msgpack unless requirements demand semantic metadata parsing. |

## What Not To Add

| Do Not Add | Reason |
|------------|--------|
| New Python dependencies such as `pyelftools`, `capstone`, `lief`, `msgpack`, or `PyYAML` | v1.17 can use external ROCm/LLVM/binutils tools and raw outputs. Add parsers only when a later phase requires structured semantic analysis. |
| A database, dataframe engine, or artifact index service | Sidecars and small raw files are enough. |
| Mandatory RGA installation | RGA is valuable but not part of the guaranteed ROCm runtime in this repo. Probe it and mark unavailable when absent. |
| Mandatory static evidence for every run | Explicitly out of milestone scope. Static evidence must be opt-in or best-effort diagnostic evidence. |
| Trace JSONL fields | Canonical traces must remain stable. Use sidecars only. |
| Score, timing, or correctness decisions based on ISA output | Static extraction can diagnose compiler artifacts, not prove runtime correctness or performance. |
| Full paper-scale static coverage | Defer unless a later requirement explicitly asks for 235-problem static artifact closure. |
| Direct replacement of PyTorch extension build with a custom `hipcc` pipeline | Too invasive for v1.17. Observe the existing build first; add custom compile paths only if later requirements need them. |
| Automatic retention of all staging directories | Can leak large/generated files and user source. Keep staging only when requested; copy/register bounded evidence artifacts. |

## Installation

No required Python dependency additions are recommended.

Existing setup remains:

```bash
uv sync --all-groups
```

Static extractors are host tools. Document them as optional system/toolchain
availability:

```bash
llvm-objdump --version
readelf --version
rga --version
roc-objdump --version
```

For richer compiler intermediates during development, users can pass saved-temp
flags through existing solution compile options rather than changing package
dependencies:

```json
{
  "compile_options": {
    "hip_cflags": ["--save-temps"]
  }
}
```

Validate exact flag behavior in the ROCm container before making saved temps a
documented default.

## Suggested Validation

Focused unit tests:

```bash
uv run pytest tests/sol_execbench/test_toolchain_routing.py
```

New v1.17 tests should cover:

- Static sidecar model round-trip and malformed payload rejection.
- Artifact discovery from fixture staging trees containing `.so`, `.hsaco`,
  `.o`, and irrelevant files.
- `llvm-objdump`, `readelf`, RGA, and `roc-objdump` command-builder behavior
  with fake runners.
- Routing fallback when primary tools are unavailable.
- Nonfatal static extraction failure during benchmark execution.
- Sidecar paths for `--output` and no-output modes.
- Guardrails that static evidence is not trace, score, correctness,
  performance, paper-parity, or leaderboard authority.

Representative smoke command after implementation:

```bash
uv run sol-execbench examples/hip_cpp/rmsnorm \
  --solution examples/hip_cpp/rmsnorm/solution.json \
  --static-evidence \
  -o out/v1.17/hip_cpp_rmsnorm.trace.jsonl \
  --keep-staging
```

Then inspect:

```bash
uv run sol-execbench toolchain --json \
  --evidence-level static \
  --artifact-type rocm_binary \
  --gpu-arch gfx1200
```

## Sources

- Repository: `.planning/PROJECT.md` - v1.17 goal, target features, and
  explicit deferrals.
- Repository: `.planning/MILESTONES.md` - v1.16 shipped routing foundation and
  v1.14/v1.15 sidecar precedent.
- Repository: `.planning/milestones/v1.16-MILESTONE-AUDIT.md` - confirms
  static capture/extraction was deferred to v1.17.
- Repository: `docs/rocm_toolchain_routing.md` - current static tool lifecycle
  entries and claim boundary.
- Repository: `src/sol_execbench/core/toolchain.py` - routing models,
  capability registry, probe behavior, and static artifact enums.
- Repository: `src/sol_execbench/driver/problem_packager.py` and
  `src/sol_execbench/driver/templates/build_ext.py` - HIP/C++ staging and
  PyTorch extension build integration.
- Repository: `src/sol_execbench/cli/main.py` - existing environment/profile
  sidecar path pattern and benchmark execution lifecycle.
- Official ROCm HIP compiler docs:
  https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html
  - HIP supports offline compilation, host/device compilation, embedded device
  code, and runtime compilation. Confidence: HIGH.
- Official ROCm compiler reference:
  https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html
  - ROCm compiler interfaces are `hipcc` and `amdclang++`; `--offload-arch`
  targets AMD GPU architectures; `offload-arch -f` can query binary offload
  requirements. Confidence: HIGH.
- Official LLVM AMDGPU usage:
  https://llvm.org/docs/AMDGPUUsage.html
  - HIP/OpenMP embed code objects; code object V5 is default when not specified;
  AMD HSA runtime requires `ET_DYN` code objects; metadata is carried in AMDGPU
  note records for v3+. Confidence: HIGH.
- Official LLVM `llvm-objdump` docs:
  https://llvm.org/docs/CommandGuide/llvm-objdump.html
  - supports object/final image inspection, headers, symbols, disassembly, and
  `--offloading`. Confidence: HIGH.
- Official GNU binutils `readelf` docs:
  https://sourceware.org/binutils/docs/binutils/readelf.html
  - supports ELF headers, sections, notes, symbols, dynamic info, and hex/string
  section dumps. Confidence: HIGH.
- GPUOpen RGA repository and manual:
  https://github.com/GPUOpen-Tools/radeon_gpu_analyzer and
  https://gpuopen.com/manuals/rga_manual/help_manual/
  - RGA can analyze precompiled AMD GPU Code Object binaries and produce ISA
  disassembly/resource information; CLI exists but exact command must be
  validated locally. Confidence: MEDIUM-HIGH.
