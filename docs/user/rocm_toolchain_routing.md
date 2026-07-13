# ROCm Toolchain Routing

ROCm tools are split across runtime, profiling, compiler, object-inspection,
GPUOpen, and migrated repository surfaces. This project treats that split as an
explicit benchmark concern. A routing decision should tell the user which tool
is available for the requested evidence, why another tool is unavailable, and
which diagnostic surface consumes the route.

## Evidence Levels

| Evidence level | Purpose | Example tools |
| --- | --- | --- |
| Runtime | Device and runtime discovery. | `rocminfo`, `rocm_agent_enumerator`, `amd-smi`, `rocm-smi` |
| Profiling | Diagnostic execution-time artifacts. | `rocprofv3`, `rocprofv3-avail` |
| Static | Diagnostic current-build artifact metadata and bounded extractor output. | RGA, `llvm-objdump`, `roc-objdump`, `readelf` |
| Derived score | Derived AMD-side score and bound artifacts. | AMD SOL/SOLAR sidecars |

Routing output is diagnostic-only metadata using the authority-class vocabulary
in `docs/user/CLAIMS.md`. It answers tool availability and provenance questions; it
does not prove benchmark correctness or performance.

## Tool Lifecycle

| Lifecycle | Meaning |
| --- | --- |
| `active` | Current tool or source used by the project. |
| `deprecated` | Historical tool retained for routing and migration notes. |
| `migrated` | Tool or repository moved to a new source of truth. |
| `planned` | Tool is intentionally deferred to a future milestone. |
| `candidate` | Tool may be used if local packaging and evidence requirements work. |
| `rejected` | Tool is documented as intentionally not used. |

The ROCm Systems super-repo is treated as an important source-of-truth signal
because it consolidates many former ROCm systems projects, including
`rocprofiler-sdk`, `rocprofiler-systems`, `rocminfo`, `amdsmi`, and related
components.

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| `available` | The registry entry matches and the bounded local probe succeeded. |
| `unavailable` | Expected executable or prerequisite was not found. |
| `unsupported_arch` | The requested GPU architecture is outside the entry's declared patterns. |
| `unsupported_artifact` | The tool does not handle the requested artifact type. |
| `deprecated` | The entry is retained for history but should not be selected. |
| `migrated` | The entry moved to a replacement or source-of-truth repository. |
| `planned` | The route is documented for future work and not selected by current evidence collection. |
| `rejected` | The tool is intentionally not used. |
| `failed` | A bounded probe ran and failed. |

## CLI

Print a routing decision:

```bash
uv run sol-execbench --format json toolchain route \
  --evidence-level profiling \
  --artifact-type executable_run \
  --gpu-arch gfx1200
```

Print the built-in registry:

```bash
uv run sol-execbench --format json toolchain list
```

Static evidence routes can be queried directly. HIP/C++ benchmark runs can also
consume active routes through `--static-evidence auto`, which writes diagnostic
sidecars documented in `docs/user/static_kernel_evidence.md`:

```bash
uv run sol-execbench --format json toolchain route \
  --evidence-level static \
  --artifact-type rocm_binary \
  --gpu-arch gfx1200
```

## Current Tool Matrix

| Tool | Lifecycle | Evidence | Notes |
| --- | --- | --- | --- |
| `rocprofv3` | active | profiling | Runtime profiling/tracing route; diagnostic only. |
| `rocprofv3-avail` | active | profiling | Counter/configuration discovery companion. |
| `rocprofiler-systems` | migrated | runtime/profiling | Historical repo; route to ROCm Systems source-of-truth metadata. |
| ROCm Systems | active | runtime/profiling | Repository source-of-truth signal, not a direct executable route. |
| `rocminfo` | active | runtime | Runtime/device discovery. |
| `rocm_agent_enumerator` | active | runtime | Architecture discovery. |
| RGA | planned | static | Rich resource extraction is deferred until local packaging and parser evidence exist. |
| `llvm-objdump` | active | static | Bounded object-inspection route for persisted HIP/C++ build artifacts. |
| `roc-objdump` | candidate | static | Distribution-dependent object-inspection candidate. |
| `readelf` | active | static | Generic ELF metadata fallback for persisted HIP/C++ build artifacts. |

## Source References

- ROCprofiler-SDK `rocprofv3` documentation:
  https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
- ROCm Systems super-repo:
  https://github.com/ROCm/rocm-systems
- Deprecated ROCprofiler Systems repository:
  https://github.com/ROCm/rocprofiler-systems
- Radeon GPU Analyzer repository:
  https://github.com/GPUOpen-Tools/radeon_gpu_analyzer
- RGA help manual:
  https://gpuopen.com/manuals/rga_manual/help_manual/
- HIP compiler documentation:
  https://rocm.docs.amd.com/projects/HIP/en/develop/understand/compilers.html
- LLVM `llvm-objdump` documentation:
  https://llvm.org/docs/CommandGuide/llvm-objdump.html

## Claim Boundary

Do not describe a successful routing decision as a correctness result,
performance proof, paper-parity result, leaderboard readiness, or hardware
validation. Static Kernel Evidence consumes routing records as diagnostic
metadata only.
