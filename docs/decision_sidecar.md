# Decision Sidecar

The Decision sidecar is optional diagnostic metadata that turns static resource
footprints plus an arch capability budget into confidence-weighted Layer R
optimization hints. It is emitted beside the trace as `<trace>.decision.json`
and never changes canonical Trace JSONL.

## Enable It

Use `--decision auto` with a normal benchmark run. Decision hints derive from
static footprints, so it must be paired with `--static-evidence auto`:

```bash
uv run sol-execbench examples/hip_cpp/rmsnorm \
  --solution examples/hip_cpp/rmsnorm/solution_hip.json \
  --static-evidence auto --decision auto \
  -o out/rmsnorm.trace.jsonl
```

The default is `--decision none`; in that mode no decision sidecar is written.

## Applicability

The sidecar requires `StaticKernelEvidenceSidecar.footprints[]`, which the
static-evidence path collects only for HIP/C++ solutions (its `is_cpp` gate)
and only when `roc-objdump` is installed (it is the footprint source). PyTorch
and Triton solutions, or HIP/C++ solutions without `roc-objdump`, produce no
footprints, so no decision sidecar is written for them.

## Output Files

| Artifact | Location |
| --- | --- |
| Canonical traces | `<trace>.jsonl` |
| Decision sidecar | `<trace>.decision.json` |

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| `available` | Static footprints and a matching arch budget were available; hints derived. |
| `partial` | Footprints were available but no arch budget matched the detected gfx (spill-only hints). |
| `unavailable` | No static footprints were available. |

## What The Sidecar Contains

The sidecar schema is `sol_execbench.decision.v1`. It contains:

- diagnostic-only authority (`authority: "diagnostic"`)
- aggregate `status` and `reason_code`
- per-hint `bottleneck_class` (Layer R closed taxonomy: `register_pressure_high`,
  `lds_pressure_high`, `spill_detected`, `workgroup_size_limited`,
  `barrier_limited`, `wavefront_misaligned`, `cache_line_misaligned`)
- `confidence` (`inferred_high` / `inferred_medium` / `inferred_low`)
- prompt-safe `recommendation` strings sourced from AMD HIP performance guidelines
- per-hint `limitations[]` and `evidence_refs[]`
- a compact `summary` (hint/footprint counts, architecture, bottleneck counts)

The static path emits Layer R (resource) signals only. Compute-bound /
memory-bound / latency-bound verdicts require runtime profiling and are never
produced from static facts.

## Claim Boundaries

The Decision sidecar is diagnostic-only static-inferred guidance, using the
authority-class vocabulary in `docs/CLAIMS.md`. It is not correctness,
performance, timing, score, paper-parity, or leaderboard authority.

Static hints are resource risk signals, most actionable for latency-bound
kernels. AMD's GPUOpen guidance is explicit that higher occupancy does not
always mean higher performance; confirm via runtime profiling before acting on
occupancy-related recommendations.

`vgpr_limit` is the architected addressing limit, not the physical register file;
derivation uses it as a static pressure proxy.

## Deferred Or Unsupported Scope

The following are not produced from static facts and remain deferred:

- Layer C (instruction-mix from disassembly statistics)
- Layer M (runtime compute-bound / memory-bound / latency-bound)
- Wavefront / cache-line alignment, workgroup-size, and barrier limits (need
  block-size or access-pattern data the static footprint does not carry)
- RDNA4 dynamic register allocation: static occupancy derivation does not hold;
  a single `inferred_low` note with an explicit limitation is emitted instead.

See `docs/decision_sidecar_contract.md` and `docs/decision-modeling-research.md`
for the contract and modeling survey.
