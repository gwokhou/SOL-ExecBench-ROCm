# Static Kernel Evidence

Static Kernel Evidence is optional diagnostic metadata for HIP/C++ solution
builds. It archives static build artifacts, runs bounded routed static
extractors when available, and writes a JSON sidecar without changing canonical
trace JSONL.

## Enable It

Use `--static-evidence auto` with a normal benchmark run:

```bash
uv run sol-execbench examples/hip_cpp/rmsnorm \
  --solution examples/hip_cpp/rmsnorm/solution_hip.json \
  --static-evidence auto \
  -o out/rmsnorm.trace.jsonl
```

The default is `--static-evidence none`; in that mode no static evidence is
collected and no static evidence sidecar is written.

## Output Files

When `--output` is provided, static evidence is written beside the trace:

| Artifact | Path |
| --- | --- |
| Canonical traces | `<trace>.jsonl` |
| Static sidecar | `<trace>.jsonl.static-evidence.json` |
| Evidence directory | `<trace>.jsonl.static-evidence/` |

Without `--output`, the fallback paths are under the temporary staging
directory: `static-evidence.json` and `static-evidence/`. Use `--output` for
durable artifacts.

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| `collected` | Build artifacts and any available routed extractor outputs were collected. |
| `partial` | Some evidence was collected, but one or more extractors failed, were unavailable, or produced incomplete output. |
| `unavailable` | The requested evidence could not be collected because required artifacts or tools were unavailable. |
| `unsupported` | The solution or artifact type does not have a stable v1.17 static evidence boundary. |
| `failed` | Optional static evidence collection or extraction failed after an attempted route. |
| `skipped` | Static evidence was not requested. |

All non-`collected` states are diagnostic. They do not change correctness,
timing, scoring, paper-parity, leaderboard, or default benchmark semantics.

## What The Sidecar Contains

The sidecar schema is `sol_execbench.static_kernel_evidence.v1`. It contains:

- diagnostic-only authority flags
- aggregate status and reason code
- persisted build artifacts such as `benchmark_kernel.so`, object files, code
  objects, or HSACO files when present
- SHA256 and size for persisted artifacts
- routed `llvm-objdump` and `readelf` tool-run records
- bounded stdout/stderr tails and bounded raw output artifact paths
- conservative metadata and disassembly presence flags
- a compact `summary` section for human-facing reporting

`llvm-objdump` and `readelf` are routed through the toolchain registry. RGA and
`roc-objdump` are not mandatory execution paths in v1.17.

## Claim Boundaries

Static Kernel Evidence is diagnostic-only static-analysis evidence, using the
authority-class vocabulary in `docs/CLAIMS.md`.

Use it to inspect what was built and what bounded static tools reported. Do not
use it to claim a kernel is correct, fast, paper-equivalent, or leaderboard
ready.

In guardrail wording: it is not correctness authority, performance authority,
timing authority, score authority, paper-parity authority, or
leaderboard authority.

## Deferred Or Unsupported Scope

The following remain unsupported, partial, or deferred unless direct evidence is
recorded in a future artifact:

- CDNA3-family live hardware validation, including MI300X (`gfx942`)
- CDNA 4 live hardware validation
- Triton ROCm cache capture
- RGA-rich resource parsing such as VGPR, SGPR, LDS, scratch, and occupancy-like
  summaries
- paper-scale static coverage for the full original benchmark denominator
- standalone static analysis of arbitrary pre-existing binaries

## CPU-Safe Coverage

The static evidence implementation has CPU-safe tests for:

- strict sidecar schema and claim-boundary fields
- current-build artifact discovery and persistence
- symlink and outside-directory guardrails
- fake routed `llvm-objdump` and `readelf` extractors
- unavailable, unsupported, failed, partial, timeout, and skipped outcomes
- CLI sidecar path and summary writing

Live GPU validation is recorded separately when the environment supports it.
