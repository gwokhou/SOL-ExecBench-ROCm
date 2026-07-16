# Static Kernel Evidence

Static Kernel Evidence is optional diagnostic metadata for HIP/C++ solution
builds. It archives static build artifacts, runs bounded routed static
extractors when available, and writes a JSON sidecar without changing canonical
trace JSONL.

## Enable It

Use `--static-evidence auto` with a normal benchmark run:

```bash
uv run sol-execbench evaluate examples/hip_cpp/rmsnorm \
  --solution examples/hip_cpp/rmsnorm/solution_hip.json \
  --static-evidence auto \
  --trace-output out/rmsnorm.trace.jsonl
```

The default is `--static-evidence none`; in that mode no static evidence is
collected and no static evidence sidecar is written.

## Output Files

When `--trace-output` is provided, static evidence is written beside the trace:

| Artifact | Location |
| --- | --- |
| Canonical traces | `<trace-output>` |
| Static sidecar | `<trace-output>.static-evidence.json` |
| Evidence directory | `<trace-output>.static-evidence/` |

Without `--trace-output`, static evidence is written under a temporary staging location
for the current invocation. Use `--trace-output` for durable, repository-independent
artifacts.

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

The sidecar schema is `sol_execbench.static_kernel_evidence.v3`. It contains:

- diagnostic-only authority flags
- aggregate status and reason code
- persisted build artifacts such as `benchmark_kernel.so`, object files, code
  objects, or HSACO files when present
- SHA256 and size for persisted artifacts
- routed `llvm-objdump`, `readelf`, and `roc-objdump` tool-run records
- bounded stdout/stderr tails and bounded raw output artifact paths
- conservative metadata and disassembly presence flags
- `isa_analyses[]` records decoded instruction counts, functional-group and
  subgroup counts, observed `mfma`/`wmma` units, code-object/disassembly
  checksums, and the exact pinned ISA specification provenance
- per-kernel `footprints[]` resource usage (VGPR/SGPR/LDS/scratch/spill/wavefront)
  parsed from routed `roc-objdump --resource-usage` output, or — on ROCm 7.x
  where `roc-objdump` is absent — from the code object's `NT_AMDGPU_METADATA`
  ELF note via a native pure-Python parser (covers CDNA + RDNA)
- a compact `summary` section for human-facing reporting

`llvm-objdump`, `readelf`, and `roc-objdump` are routed through the toolchain
registry. In `auto` mode the AMD ISA integration also extracts the exact target
code object from a HIP fat binary, disassembles it, and decodes it using AMD's
machine-readable XML. Its helper and pinned XML are installed into the user
cache on first use. If the helper, XML, network, or ROCm fat-binary tools are
unavailable, collection remains diagnostic and the aggregate status becomes
`partial` rather than failing the benchmark. Set
`SOL_EXECBENCH_AMD_ISA_OFFLINE=1` to forbid ISA downloads. `rga` remains a
planned route and is not yet wired as an extractor.

## Claim Boundaries

Static Kernel Evidence is diagnostic-only static-analysis evidence, using the
authority-class vocabulary in `docs/user/CLAIMS.md`.

Use it to inspect what was built and what bounded static tools reported. Do not
use it to claim a kernel is correct, fast, paper-equivalent, or leaderboard
ready.

In guardrail wording: it is not correctness authority, performance authority,
timing authority, score authority, paper-parity authority, or
leaderboard authority.

The `footprints[]` resource-usage fields are raw facts only. Bottleneck
classification and optimization recommendations belong to a separate Decision
sidecar (see `docs/user/decision_sidecar_contract.md`) that consumes these facts;
this sidecar does not carry decision fields.

## Deferred Or Unsupported Scope

The following remain unsupported, partial, or deferred unless direct evidence is
recorded in a future artifact:

- CDNA3-family live hardware validation, including MI300X (`gfx942`)
- CDNA 4 live hardware validation
- Triton ROCm cache capture
- `rga`-rich resource parsing beyond what `roc-objdump --resource-usage` reports (RGA-rich resource parsing itself remains deferred)
- paper-scale static coverage for the full original benchmark denominator
- standalone static analysis of arbitrary pre-existing binaries

Per-kernel VGPR, SGPR, LDS, scratch, spill, and wavefront-size fields are
collected into `footprints[]` from routed `roc-objdump` output, or — on ROCm 7.x
(roc-objdump removed) — from the code object's `NT_AMDGPU_METADATA` ELF note via
a native pure-Python msgpack parser. `occupancy_estimate_waves_per_cu` stays
`null` on the metadata path (the note does not carry occupancy); the decision
layer derives resource pressure from the vgpr/limit ratio instead. Uncovered
fields stay `null` rather than being speculated.

## CPU-Safe Coverage

The static evidence implementation has CPU-safe tests for:

- strict sidecar schema and claim-boundary fields
- current-build artifact discovery and persistence
- symlink and outside-directory guardrails
- fake routed `llvm-objdump` and `readelf` extractors
- unavailable, unsupported, failed, partial, timeout, and skipped outcomes
- CLI sidecar path and summary writing
- machine-readable ISA aggregation and soft-unavailable behavior

Live GPU validation is recorded separately when the environment supports it.
