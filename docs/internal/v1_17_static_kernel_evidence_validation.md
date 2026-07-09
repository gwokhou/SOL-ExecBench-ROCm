# v1.17 Static Kernel Evidence Validation

**Date:** 2026-05-26  
**Scope:** bounded RDNA 4 static evidence validation  
**Status:** static evidence collected; benchmark correctness did not pass

## Environment Probe

| Check | Result |
| --- | --- |
| `hipcc` | available at `/usr/bin/hipcc` |
| `rocminfo` | available at `/usr/bin/rocminfo` |
| GPU target | RDNA 4 `gfx1200` detected by `rocminfo` |
| `rocm-smi` | detected one AMD GPU device |

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run sol-execbench examples/hip_cpp/rmsnorm \
  --solution examples/hip_cpp/rmsnorm/solution_hip.json \
  --static-evidence auto \
  -o <out>/rdna4-static-evidence/rmsnorm.trace.jsonl \
  --timeout 120 \
  --compile-timeout 120
```

## Static Evidence Result

The run compiled the HIP/C++ solution and wrote:

- a trace-adjacent static-evidence JSON sidecar:
  `out/evidence/legacy-artifacts/artifacts-root/v1.17/rdna4-static-evidence/rmsnorm.trace.jsonl.static-evidence.json`
- a trace-adjacent static-evidence artifact directory

The static evidence sidecar reported:

| Field | Value |
| --- | --- |
| status | `collected` |
| reason code | `static_evidence_collected` |
| artifact count | 9 |
| tool-run count | 6 |
| metadata present | true |
| disassembly present | true |
| failed tool runs | 0 |
| unsupported tool runs | 0 |

The archived evidence included `benchmark_kernel.so`, object files, and bounded raw
`llvm-objdump` / `readelf` outputs for each inspectable artifact.

## Benchmark Result Boundary

The benchmark command exited nonzero because all 14 workloads returned
`RUNTIME_ERROR` with `hidden_states must be a HIP tensor`. This validation
artifact therefore proves only that the RDNA 4 environment can build a HIP/C++
example and collect v1.17 static evidence for it. This artifact does not claim benchmark correctness.
It also does not claim timing validity, performance, score validity, paper parity, or leaderboard
readiness.

## Deferred Scope

This artifact does not validate CDNA 3, CDNA 4, Triton cache capture, RGA-rich
resource parsing, or paper-scale static coverage.
