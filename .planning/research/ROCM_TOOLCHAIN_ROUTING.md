# ROCm Toolchain Routing Research

**Milestone:** v1.16 ROCm Toolchain Research and Capability Routing
**Date:** 2026-05-25

## Research Question

ROCm exposes profiling, runtime, compiler, object, and GPUOpen analysis
capabilities across several repositories and tools. The benchmark needs a
central routing layer that can decide which tool is usable for a given hardware
generation, GPU architecture, ROCm version, artifact type, and evidence level.

Static Kernel Evidence is intentionally deferred to v1.17. v1.16 only creates
the research-backed map, lifecycle model, registry schema, routing policy, and
documentation/guardrails needed before static extractors are attached.

## Primary Sources Reviewed

| Source | What It Contributes |
| --- | --- |
| ROCprofiler-SDK `rocprofv3` docs | `rocprofv3` tracing/profiling options, output formats, configuration output, kernel filtering, counter collection, and install assumptions. |
| ROCm Systems super-repo | Current migration source of truth for many ROCm systems projects, including `rocprofiler`, `rocprofiler-sdk`, `rocprofiler-systems`, `rocminfo`, `amdsmi`, and related components. |
| Deprecated `rocprofiler-systems` repository | Historical lifecycle signal: old repositories can become deprecated and moved, so registry entries need lifecycle and replacement fields. |
| RGA repository and GPUOpen manual | RGA is an offline compiler/code-analysis tool with ISA disassembly, resource usage, and compiled binary outputs, but its domain and supported input modes differ from runtime profilers. |
| HIP compiler docs | HIP compiler/offloading documentation points to ROCm binary metadata, kernel symbols, target architectures, and linkage details as available object-level information. |
| LLVM `llvm-objdump` docs | Generic object-file dumper capabilities that can serve as a fallback or baseline object inspection tool. |

## Findings

### Toolchain fragmentation is a first-class benchmark concern

The ROCm Systems repository describes itself as a super-repo consolidating
multiple ROCm systems projects and marks many formerly separate components as
migrated or source-of-truth changed. This means a benchmark cannot rely only on
old repository names or package names. Tool metadata needs lifecycle fields:
`active`, `deprecated`, `migrated`, `planned`, `rejected`, and replacement
guidance.

### Profiling tools and static tools have different evidence authority

`rocprofv3` produces tracing, profiling, counter, summary, and output-format
evidence for executed programs. RGA and object tools are compiler/static
inspection surfaces. They must share routing status vocabulary but remain
separate evidence levels: runtime, profiling, static, and derived-score.

### Registry entries need both static facts and dynamic probes

Static registry facts can express known lifecycle, supported evidence levels,
expected binaries, and known artifact classes. Dynamic probes must record what
is actually present on the host: executable path, version command result, ROCm
root, detected GPU arch, and whether a dry-run/list command succeeds.

### Artifact type is as important as GPU architecture

A tool can be installed and still not support the specific artifact. v1.16 must
model artifact classes such as executable-run profiling, ROCm binary/code
object, ELF object, HIP compiler output, and future HSACO/static artifacts.

### Static evidence depends on routing but is a later milestone

v1.17 can add code-object capture, ISA extraction, and
`static_kernel_evidence.v1`. v1.16 must stop at defining the routing contract
and matrix so static extractors have a stable compatibility layer.

## Candidate Tool Inventory

| Tool | Lifecycle | Evidence Level | Initial Routing Notes |
| --- | --- | --- | --- |
| `rocprofv3` | active | profiling | Probe executable, version, output-format support, counter availability, and target process support. |
| `rocprofv3-avail` | active | profiling | Probe available counters/configurations when installed with ROCprofiler-SDK. |
| ROCm Systems `rocprofiler-sdk` | active/migrated | profiling | Treat super-repo as source-of-truth signal for future lifecycle updates. |
| `rocprofiler-systems` old repo | deprecated/migrated | profiling/runtime | Keep as historical alias with replacement target `ROCm/rocm-systems`. |
| RGA | active external | static/future | Planned for v1.17 static evidence; v1.16 records lifecycle and expected capability boundaries. |
| `llvm-objdump` | active external | static/future | Planned baseline object inspection; model availability separately from ROCm packages. |
| `roc-objdump` | candidate | static/future | Probe by executable presence; document if absent or distribution-specific. |
| `readelf` | active external | static/future | Generic ELF metadata fallback, not AMD-specific performance evidence. |
| `rocm_agent_enumerator` | active/candidate | runtime | Useful for arch discovery; route separately from evidence extraction. |
| `rocminfo` | active | runtime | Runtime/device discovery; useful for arch and agent metadata. |
| `amd-smi` / `rocm-smi` | active/runtime variant | runtime | Device/runtime context, not compiler evidence. |

## Required Registry Fields

- `tool_id`
- `display_name`
- `lifecycle`
- `replacement_tool_id`
- `evidence_levels`
- `artifact_types`
- `hardware_generations`
- `gpu_arch_patterns`
- `rocm_version_min`
- `rocm_version_max`
- `expected_binaries`
- `probe_commands`
- `status`
- `reason_code`
- `source_refs`

## Required Status Vocabulary

- `available`
- `unavailable`
- `unsupported_arch`
- `unsupported_artifact`
- `deprecated`
- `migrated`
- `planned`
- `rejected`
- `failed`

## Sources

- ROCprofiler-SDK `rocprofv3` docs:
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- ROCm Systems super-repo:
  https://github.com/ROCm/rocm-systems
- Deprecated ROCprofiler Systems repo:
  https://github.com/ROCm/rocprofiler-systems
- Radeon GPU Analyzer repo:
  https://github.com/GPUOpen-Tools/radeon_gpu_analyzer
- RGA help manual:
  https://gpuopen.com/manuals/rga_manual/help_manual/
- HIP compiler docs:
  https://rocm.docs.amd.com/projects/HIP/en/develop/understand/compilers.html
- LLVM `llvm-objdump` docs:
  https://llvm.org/docs/CommandGuide/llvm-objdump.html
