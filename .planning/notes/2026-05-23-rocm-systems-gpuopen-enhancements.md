---
date: "2026-05-23 20:09"
promoted: false
---

# ROCm Systems and GPUOpen enhancement research

## Context

This note captures two enhancement research passes for the SOL ExecBench ROCm port:

- ROCm Systems: `https://github.com/ROCm/rocm-systems`
- GPUOpen: GPUOpen manuals, tools, and GitHub repositories

The project goal is to evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving SOL ExecBench semantics. These sources should be used as optional data and infrastructure enhancements, not as replacement benchmark datasets.

## Executive summary

ROCm Systems is the stronger source for runtime infrastructure: profiling, telemetry, ROCm environment checks, HIP smoke tests, and CDNA-oriented performance analysis.

GPUOpen is the stronger source for static kernel analysis and RDNA/Radeon-side tooling: machine-readable ISA, Radeon GPU Analyzer, RDNA counter taxonomy, device mapping, and manual debugging tools.

Recommended near-term direction:

1. Add hardware/environment fingerprinting with `amd-smi` and `rocminfo`.
2. Add optional `rocprofv3` profiling evidence, preferably `rocpd` with CSV conversion as needed.
3. Add static kernel/code-object analysis using RGA or equivalent ROCm tools.
4. Use GPUOpen machine-readable ISA XML as an instruction classification dictionary for RDNA4 and CDNA3.
5. Keep RGP/RGD and GPUOpen graphics tools as manual diagnostics, not default CI or scoring dependencies.

## ROCm Systems findings

### High-value components

| Priority | Component | Project value |
|---|---|---|
| P0 | `rocprofiler-sdk` / `rocprofv3` | Standard profiling evidence for HIP/HSA/kernel/memory/RCCL traces |
| P0 | `amdsmi` and `rocminfo` | Environment fingerprinting, GPU identity, version capture, telemetry |
| P1 | `rocprofiler-compute` | CDNA/MI SOL, roofline, counter-derived analysis |
| P1 | `hip-tests` | HIP build/runtime smoke tests for Docker and GPU environments |
| P2 | `rdc` | Cluster/job telemetry, Prometheus/Grafana, RAS/ECC monitoring |
| P2 | super-repo CI/Docker patterns | Sparse checkout, ROCm version matrix, profiler container references |

### Recommended data additions

Add a run-level hardware snapshot:

```json
{
  "hardware": {
    "gpu_name": "MI300X",
    "gfx_target": "gfx942",
    "compute_units": 304,
    "memory_total_bytes": 201000000000,
    "rocm_version": "7.x",
    "driver_version": "...",
    "hip_visible_devices": "0"
  }
}
```

Add optional profiling artifacts:

```json
{
  "profile": {
    "tool": "rocprofv3",
    "formats": ["rocpd", "csv"],
    "artifacts": {
      "results_db": "..._results.db",
      "kernel_trace_csv": "..._kernel_trace.csv",
      "counter_collection_csv": "..._counter_collection.csv",
      "agent_info_csv": "..._agent_info.csv"
    },
    "counters": ["GPU_UTIL", "MfmaUtil", "BANDWIDTH_EA"]
  }
}
```

Add optional CDNA analysis:

```json
{
  "analysis": {
    "speed_of_light": {
      "valu_util": 0.72,
      "mfma_util": 0.64,
      "hbm_bw_pct": 0.51
    },
    "roofline": {
      "ai_hbm": 12.3,
      "gflops": 52400,
      "bound": "memory"
    }
  }
}
```

### Design guidance

- Prefer `rocprofv3` over old `rocprof`, `ROCTracer`, and ROCprofiler v1/v2 paths.
- Prefer `rocpd` as the persisted profile artifact because it is a SQLite database and can be converted to CSV, PFTrace, or OTF2.
- Treat `rocprofiler-compute` as optional, especially because it targets MI/CDNA accelerators more directly than RDNA consumer GPUs.
- Use `hip-tests` only as inspiration for lightweight smoke tests, not as a copied test suite.
- Use RDC only if the project grows into multi-node or long-running benchmark infrastructure.

### Candidate implementation work

- Add `sol-execbench env-snapshot` or equivalent internal utility that gathers `amd-smi`, `rocminfo`, `rocm_agent_enumerator`, and PyTorch ROCm metadata.
- Add `--profile rocprofv3` to the evaluation driver.
- Add artifact schema fields for `rocpd`, CSV conversion outputs, and profiler command provenance.
- Add pytest markers that auto-skip based on detected `gfx` target: `requires_rdna4`, `requires_cdna3`.
- Add small ROCm smoke tests for `hipcc`, HIP runtime init, simple kernel launch, device memory copy, and event timing.

## GPUOpen findings

### High-value components

| Priority | Component | Project value |
|---|---|---|
| P0 | Machine-readable AMD GPU ISA XML | Instruction classification for RDNA4/CDNA3 static analysis |
| P0 | Radeon GPU Analyzer (RGA) | Code-object analysis: ISA, VGPR/SGPR, LDS, scratch, CFG |
| P1 | GPU Performance API counter docs | RDNA counter taxonomy and explanations |
| P1 | `GPUOpen-Tools/device_info` | Device ID / product / GFX IP mapping reference |
| P2 | Radeon GPU Profiler (RGP) | Manual RDNA/HIP Windows-side deep profiling |
| P3 | Radeon GPU Detective (RGD) | Windows DX12/Vulkan crash dump analysis; low project relevance |

### Recommended data additions

Add static ISA features:

```json
{
  "static_isa": {
    "arch": "gfx942",
    "instruction_counts": {
      "valu": 123,
      "salu": 41,
      "vmem": 16,
      "lds": 8,
      "mfma": 12
    },
    "uses_matrix_core": true,
    "uses_lds": true,
    "estimated_memory_intensity": "high"
  }
}
```

Add static code-object resource features:

```json
{
  "static_analysis": {
    "tool": "rga",
    "vgpr_count": 96,
    "sgpr_count": 48,
    "lds_bytes": 8192,
    "scratch_bytes": 0,
    "occupancy_hint": "register-limited",
    "code_object_analyzed": true
  }
}
```

Add device mapping features:

```json
{
  "device": {
    "pci_id": "0x7550",
    "marketing_name": "Radeon RX 9070 XT",
    "gfx_ip": "gfx1201",
    "family": "RDNA4"
  }
}
```

### Design guidance

- Use GPUOpen machine-readable ISA XML as a data dictionary, not as a runtime dependency.
- Use RGA or equivalent binary analysis as an optional post-build diagnostic path.
- Keep RGA output out of correctness scoring by default; use it for evidence and report enrichment.
- Use GPU Performance API documentation to explain RDNA counters, but do not make GPA the default runtime collector for ROCm/HIP benchmarks.
- Do not prioritize Radeon GPU Detective because it targets Windows DX12/Vulkan crash workflows.
- Do not prioritize graphics-focused GPUOpen SDKs such as FidelityFX, RMV, or RRA for this ROCm benchmark port.

### Candidate implementation work

- Add `sol-execbench analyze-static --solution solution.json` or equivalent internal analyzer.
- Add a normalized static-analysis schema for VGPR, SGPR, LDS, scratch, ISA mix, and matrix-core usage.
- Add an architecture dictionary sourced from GPUOpen machine-readable ISA for `RDNA4` and `CDNA3`.
- Add docs explaining how to use RGA/RGP manually when investigating anomalous benchmark results.
- Add a counter glossary that maps project report metrics to ROCm profiler and GPUOpen terminology.

## Combined roadmap

### Phase candidate: runtime evidence

Goal: make every benchmark run reproducible and diagnosable.

Work:

- Capture `amd-smi` and `rocminfo` environment metadata.
- Persist ROCm version, driver version, GPU identity, and `gfx` target.
- Add optional `rocprofv3` artifact collection.
- Preserve profiler command line and output paths in result metadata.

Acceptance:

- Benchmark results include hardware and ROCm environment fingerprints.
- If `rocprofv3` is available and enabled, result artifacts include `rocpd` or CSV traces.
- If profiling tools are missing, benchmark still runs with a clear skipped-evidence status.

### Phase candidate: static kernel evidence

Goal: enrich solution reports with code-object and ISA-level evidence.

Work:

- Extract or locate compiled AMD GPU code objects.
- Run RGA/binary analysis when available.
- Parse resource usage: VGPR, SGPR, LDS, scratch.
- Classify ISA instructions using GPUOpen ISA data.

Acceptance:

- Reports can explain register-limited, LDS-heavy, scratch-using, memory-heavy, and MFMA-using kernels.
- Static analysis is optional and never blocks correctness evaluation.
- RDNA4 and CDNA3 are explicitly supported in the classification layer.

### Phase candidate: environment smoke tests

Goal: fail fast on broken ROCm/Docker/GPU setups.

Work:

- Add lightweight HIP smoke tests inspired by `hip-tests`.
- Test `hipcc`, HIP runtime init, device memory copy, simple kernel launch, and event timing.
- Auto-skip architecture-specific tests based on detected `gfx` target.

Acceptance:

- A developer can quickly distinguish project failures from ROCm installation failures.
- Docker/GPU environment checks are documented and runnable independently of full benchmark execution.

## Non-goals

- Do not import `rocm-systems` as a submodule.
- Do not copy large ROCm or GPUOpen source trees into this repository.
- Do not use GPUOpen graphics tooling as a default Linux ROCm benchmark dependency.
- Do not change public SOL ExecBench correctness semantics based on profiler or static-analysis metrics.
- Do not make CDNA-only profiler-compute features mandatory for RDNA4.

## References

- ROCm Systems: `https://github.com/ROCm/rocm-systems`
- ROCprofiler-SDK / rocprofv3: `https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk`
- ROCm Compute Profiler: `https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute`
- AMD SMI: `https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi`
- rocminfo: `https://github.com/ROCm/rocm-systems/tree/develop/projects/rocminfo`
- HIP tests: `https://github.com/ROCm/rocm-systems/tree/develop/projects/hip-tests`
- GPUOpen machine-readable ISA: `https://gpuopen.com/machine-readable-isa/`
- Radeon GPU Analyzer: `https://github.com/GPUOpen-Tools/radeon_gpu_analyzer`
- GPU Performance API: `https://github.com/GPUOpen-Tools/gpu_performance_api`
- Device info: `https://github.com/GPUOpen-Tools/device_info`
- Radeon GPU Profiler: `https://github.com/GPUOpen-Tools/radeon_gpu_profiler`
