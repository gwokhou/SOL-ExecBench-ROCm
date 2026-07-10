# Decision Sidecar Modeling Research

> Research survey for the future **Decision sidecar** (`sol_execbench.decision.v1`,
> not yet implemented). This is the companion to
> `docs/decision_sidecar_contract.md` (the mount-point contract). It records the
> bottleneck-taxonomy sources, the static-vs-runtime inference boundary, the
> proposed decision model, and **source-credibility assessments** for the
> materials reviewed. It is a research artifact; it carries diagnostic guidance
> only and never asserts benchmark authority.

## 1. Purpose and Scope

The data layer (environment capability budgets + static resource footprints)
carries raw, neutrally-named facts. The Decision sidecar will turn those facts
into structured, confidence-weighted optimization hints. Before modeling that
layer, this survey answers three questions:

1. What bottleneck taxonomy does AMD publish, and which parts of it can be
   inferred from **static** artifacts (the data we actually collect) vs. which
   require **runtime** profiling?
2. How do the two covered architectures — CDNA3 (`gfx942`) and RDNA 3.5
   (`gfx1150`) — diverge in ways that force the decision model to branch?
3. Which external sources are trustworthy enough to cite, and which must be
   rejected or down-weighted?

**Out of scope**: implementing the decision schema or run path. Those belong to
a later `/gsd-plan-phase` workflow that consumes this document.

## 2. Source Credibility Hierarchy

The survey follows a strict credibility order: **AMD official documentation >
AMD-related research papers > NVIDIA official = NVIDIA research papers** (the
last tier is used only for cross-validation when AMD material is thin).

In practice, AMD official material was sufficient for every modeling decision.
NVIDIA material (Nsight Compute SOL taxonomy) was used only to confirm that the
compute/memory/latency three-way split is cross-vendor consensus. Two
third-party materials provided mid-survey are assessed in §10; one is retained
as a methodology reference, the other is rejected after sampling.

## 3. AMD Official Bottleneck Taxonomy (Omniperf)

Omniperf (rocprofiler-compute), AMD's official profiler, organizes its analysis
into **18 panel categories**, each with a Speed-of-Light (SOL) view that reports
utilization as a percentage of theoretical peak. These collapse into three
resource dimensions that form a closed taxonomy:

| Dimension | Representative panels |
| --- | --- |
| **Compute / Instruction** | System SOL; CU Instruction Mix (VALU/MFMA/VMEM); CU Compute Pipeline SOL |
| **Memory hierarchy** | Memory Chart; Roofline; L1/L2/Constant/Instruction Cache SOL + Stalls; Memory Latencies |
| **Resource / Scheduling** | Wavefront Occupancy; Wavefront Lifetime/Launch; SPI Resource Allocation; LDS SOL |

Two AMD conventions feed directly into the decision model:

- **Color thresholds**: _yellow > 50%_, _red > 90%_ of peak. These become the
  `confidence` encoding (§8.3).
- **"Top bottleneck kernels/dispatches"** panels: Omniperf natively ranks
  kernels by total duration, confirming that severity-ordered output is the
  vendor's own convention.

## 4. RDNA vs CDNA Structural Divergence

The two covered architectures diverge on every axis the occupancy formula
depends on. The official reference is the HIP _Hardware Implementation_ chapter.

| Axis | CDNA3 (`gfx942`) | RDNA 3.5 (`gfx1150`) | RDNA 4 (`gfx1200`, not yet covered) |
| --- | --- | --- | --- |
| Wavefront | 64 (wave64) | 32 (wave64 optional) | 32 |
| Base unit | CU | **WGP** (2 CUs) | WGP |
| SIMD org | 4 × SIMD64 per CU | **2 × SIMD32 per CU** | 2 × SIMD32 per CU |
| VGPR allocation | static (granularity 16) | static | **dynamic** (fundamental change) |
| Physical VGPR / SIMD | 512 / EU | ~1536 / SIMD (RDNA3 class) | 512 KiB / SIMD |
| LDS banks / bandwidth | 32 banks, 128 B/cycle | **64 banks, 256 B/cycle** | 64 banks, 256 B/cycle |
| LDS scope | CU | **WGP** (threadgroup must fit one WGP) | WGP |
| Cache line | 64 B | **128 B** (32 threads × 4 B) | 128 B |
| Matrix unit | **MFMA** | **WMMA** | WMMA |

**Implication**: the CDNA3 closed-form occupancy formula cannot be transplanted
to RDNA unchanged. The divergence axes (matrix unit, register-allocation model,
CU/WGP grouping) are exactly what the `ArchIsaBudget` schema now exposes so the
decision layer can select the right formula tier (§8, §9).

## 5. Static-Inferable vs Runtime-Only Boundary

This is the core constraint. The data layer's inputs are **static** (roc-objdump
footprint + arch ISA budget); most Omniperf SOL panels need **runtime** hardware
counters.

| Bottleneck class | Static-inferable? | Basis |
| --- | --- | --- |
| VGPR-limited occupancy | yes (exact) | MI300X doc closed-form formula |
| LDS-limited occupancy | yes (exact) | MI300X doc closed-form formula |
| Register spilling | yes (deterministic) | `scratch_bytes > 0` ⟹ `spill_detected` |
| Wavefront / block alignment | yes (heuristic) | wavefront_size vs. block multiplicity (64 / 32) |
| WorkGroup-mapping alignment | yes (heuristic) | MI300X: multiples of XCD count (= 8) |
| Compute-bound vs memory-bound | no — runtime | needs arithmetic intensity + measured bandwidth |
| Cache hit rate / stall reasons | no — runtime | needs L1/L2 counters |
| Instruction mix (VALU/MFMA share) | partial — needs disassembly statistics, not resource footprint |

**Consequence**: the static decision path can only emit **resource-dimension
risk signals**, never a compute/memory-bound verdict. This matches the
mount-point contract's precedence rule: _runtime measured > static inferred;
static informs the pre-run and no-profile fallback cases_. The static path is a
**risk-hint layer**, not a measurement layer.

## 6. Occupancy Derivation Rules

### 6.1 CDNA3 (`gfx942`) — closed form (MI300X Workload Optimization doc)

```
inputs (data layer):
  footprint.vgpr_used          (roc-objdump NumVgprs)
  footprint.lds_bytes          (LDSByteSize)
  budget.vgpr_physical, budget.lds_per_workgroup_bytes, budget.simd_per_cu
  nW = waves_per_workgroup     (GAP — not in static footprint; see §9)

derive:
  occ_vgpr = lookup_by_vgpr(vgpr_used, budget)     # round up to 16-granularity, invert to waves/EU
  occ_lds  = floor(budget.lds_per_workgroup_bytes / lds_bytes)
  occ      = min(floor(occ_vgpr * simd_per_cu / nW), occ_lds) * nW / simd_per_cu
```

### 6.2 RDNA 3.5 (`gfx1150`) — static (GPUOpen "Occupancy Explained")

```
occupancy = assigned_waves / wave_slots_per_simd        # 16 slots/SIMD on RDNA2+
limited by min(VGPR, LDS, Thread Group Size, Barriers)  # official 4-way limiter
WGP scope: a threadgroup must reside on a single WGP (shared LDS)
```

### 6.3 Vendor-neutral form (cross-validated)

```
O = floor(F / (R * W * w))     # F = register file per CU; R = regs/thread; W = wave size
```

This is equivalent to the CDNA3 form on the VGPR axis and matches the
Instruction Roofline Model formulation.

### 6.4 RDNA 4 (`gfx1200`) — dynamic allocation defeats static derivation

RDNA 4 introduces **dynamic VGPR allocation** (Hot Chips 2025; Chips and Cheese
analysis). Static occupancy derivation no longer holds. The decision path must
fall back to the `Occupancy` value roc-objdump reports directly and downgrade
confidence to `inferred_low`, recording the limitation explicitly. This is why
`register_allocation_model` is a first-class budget field (§9).

## 7. Official Caveat: Occupancy != Performance

AMD's GPUOpen _Occupancy Explained_ states this unambiguously, and it is the
single most important input to the confidence model:

- _"Does better occupancy mean better performance? **No**."_
- Only **latency-bound** workloads may benefit from higher occupancy.
- In **memory-bound** regimes, raising occupancy can **reduce** performance via
  cache thrashing.
- Peak occupancy does not always mean peak performance.

**Modeling consequence**: an `OCCUPANCY_LOW` hint must **not** recommend "raise
occupancy" — that is often useless or harmful. The recommendation must instead
be _"occupancy may limit latency hiding; profile to confirm the kernel is
latency-bound before trading occupancy for ILP or per-wave work."_ Except for
deterministic facts (`spill_detected`, a granularity-boundary occupancy jump),
occupancy signals default to `inferred_low`, because static analysis cannot tell
whether the kernel is latency-bound.

## 8. Proposed Decision Model

### 8.1 Bottleneck taxonomy (closed enum, three layers)

```
Layer R — Resource (static-inferable; backed by the current data layer)
  REGISTER_PRESSURE_HIGH        # "Limited by VGPR"
  LDS_PRESSURE_HIGH             # "Limited by LDS"
  WORKGROUP_SIZE_LIMITED        # "Limited by Thread Group Size"
  BARRIER_LIMITED               # "Limited by Barriers" (rare)
  SPILL_DETECTED                # scratch > 0 (deterministic)
  WAVEFRONT_MISALIGNED          # block not a multiple of wavefront size
  CACHE_LINE_MISALIGNED         # RDNA 128 B / CDNA 64 B coalescing risk
Layer C — Compile-time (disassembly-derived; partially deferred)
  INSTRUCTION_MIX_SKEW          # low MFMA/WMMA or high scalar share
Layer M — Runtime measured (injected from profile_summary.v2; static path never emits)
  COMPUTE_BOUND / MEMORY_BOUND / LATENCY_BOUND
```

The static path emits **Layer R only**; everything else stays `null` rather than
being promoted — the same "unknown → null, never promote" discipline as the data
layer, now with AMD-official backing.

### 8.2 Recommendation templates (from HIP Performance Guidelines)

| Bottleneck class | Recommendation source (AMD HIP guidelines) |
| --- | --- |
| REGISTER_PRESSURE_HIGH | minimize live variables; chain function calls; move per-thread temp arrays to LDS; `__launch_bounds__(block, min_blocks_per_cu)` |
| LDS_PRESSURE_HIGH | reduce per-block LDS; dynamic LDS allocation; trim tile size |
| SPILL_DETECTED | lower VGPR (`waves_per_eu` hint); reorder compute graph to shorten live ranges |
| OCCUPANCY_LOW | confirm latency-bound via profile before acting (§7); consider ILP if already maxed |
| WAVEFRONT_MISALIGNED | block size a multiple of wavefront_size (CDNA = 64, RDNA = 32) |

### 8.3 Confidence encoding (borrows AMD SOL color rules)

```
inferred_high    # deterministic: spill_detected; occupancy jump at a granularity boundary
inferred_medium  # ratio threshold: occupancy < 50% of waves_per_cu_max (AMD yellow)
inferred_low     # boundary-adjacent or needs runtime confirmation (AMD red zone; non-latency-bound unknown)
```

The `inferred_*` prefix distinguishes static hints from `measured_*` runtime
values in `profile_summary.v2`, realizing the precedence rule.

### 8.4 Precedence

```
runtime measured (profile_summary.v2)  >  static inferred (footprint + budget)  >  no data
on conflict, the static hint is demoted into limitations[] and never overrides runtime.
```

## 9. Data Layer Alignment

The data layer (commits on `feat/decision-ready-data-layer`) now exposes the
fields the decision formulas need. `ArchIsaBudget` (`arch_capability_budget.v1`)
carries 18 keys, organized by the invariant/dialect/divergence split:

| Tier | Fields |
| --- | --- |
| invariant | `wavefront_size`, `lds_per_workgroup_bytes`, `register_file_per_cu_bytes` |
| dialect (parameterized) | `simd_per_cu`, `wave_slots_per_simd`, `cache_line_bytes`, `vgpr_limit`, `sgpr_limit`, `waves_per_cu_max`, `supported_dtypes` |
| divergence (routing) | `matrix_unit` (`mfma`/`wmma`/`none`), `register_allocation_model` (`static`/`dynamic`), `compute_unit_grouping` (`cu`/`wgp`) |

Current populated values:

| Field | gfx942 (CDNA3) | gfx1150 (RDNA 3.5) |
| --- | --- | --- |
| `matrix_unit` | `mfma` | `wmma` |
| `register_allocation_model` | `static` | `static` |
| `compute_unit_grouping` | `cu` | `wgp` |
| `simd_per_cu` | 4 | 2 |
| `wave_slots_per_simd` | 10 | 16 |
| `waves_per_cu_max` | 40 | 32 |
| `cache_line_bytes` | `null` (no clear primary source) | 128 |
| `register_file_per_cu_bytes` | 524288 (VGPR+AGPR combined) | `null` (no reliable primary source) |
| `lds_per_workgroup_bytes` | 65536 | `null` (RDNA3.5 LDS/L1 split unconfirmed) |

**Known gap (deferred to the decision modeling workflow)**: the closed-form
occupancy formula needs `nW` (waves per workgroup = block size / wavefront
size), which is not in the static footprint. Three resolution options, in
ascending cost: (a) use the `Occupancy` value roc-objdump reports directly and
skip the formula; (b) extend the footprint with workgroup size from disassembly;
(c) declare precise occupancy a deferred limitation and emit only spill,
granularity-boundary, and wavefront-alignment signals.

**Semantic note**: `vgpr_limit` is the architected *addressing* limit (256), not
the physical register file. The formula needs the physical file
(`register_file_per_cu_bytes`). The two are distinct dimensions; conflating them
produces wrong occupancy. This is recorded for the decision workflow.

## 10. Third-Party Material Assessment

Two materials were supplied mid-survey for cross-vendor / data coverage.

### 10.1 arXiv:2603.28793 — "Toward a Universal GPU ISA"

**Retention**: methodology reference only; not a fact source.

Credibility red flags (documented so the decision workflow does not over-rely
on it):

- Authors are not known GPU-architecture researchers; the arXiv record lists no
  affiliation.
- The paper discloses generative-AI assistance (Claude) for LaTeX/diagrams.
- The workload claim (~5000 primary-source pages; "~4800 pages" of AMD ISA
  guides alone) is implausible for a 7-page preprint.
- AMD and Intel benchmarks are "in progress"; the taxonomy is validated only on
  NVIDIA T4 and Apple M1.
- Self-inconsistency: the abstract, body, and conclusion disagree on whether
  there are ten or eleven invariant primitives.

Usable contributions (each must be re-verified against AMD official sources):

- The **invariant / dialect / divergence** three-way split — a more explanatory
  organization than compile-time/runtime, and now reflected in the budget schema
  (§9).
- The vendor-neutral occupancy form `O = floor(F / (R * W * w))` (§6.3),
  equivalent to the CDNA3 derivation.
- Physical-rationale arguments (lockstep groups exist because instruction fetch
  costs 10–100× a single-lane arithmetic op; zero-cost switching because memory
  latency is 100–800 cycles) — consistent with AMD GPUOpen material.
- The AMD mapping (wavefronts W=32/64, VGPR, LDS, EXEC mask, `S_WAITCNT`, SALU
  hoist) is accurate.

### 10.2 RightNow-AI/RightNow-GPU-Database — **excluded**

A 2,824-GPU product-specification database (AMD/NVIDIA/Intel), Apache-2.0, whose
data originates from TechPowerUp via `dbgpu`. Sampled against the AMD dataset
(`data/amd/all.json`, 1,292 entries). **Verdict: not used.** Five independent
reasons:

1. **No gfx-target identifier.** `gfx942` and `gfx1150` return zero hits; the
   set is keyed by product name and architecture label, not ISA target. The
   project's data layer is gfx-keyed; no mapping exists.
2. **AMD entries lack the occupancy-critical fields.** AMD records carry only
   `wavefrontSize`, `maxWorkGroupSize`, `ldsPerCU`, `computeUnits`. There is no
   register-file field (no AMD equivalent of NVIDIA's `registersPerSM`), no
   VGPR granularity, no wave-slot count, no SIMD-per-CU, no cache-line size, no
   matrix-unit type.
3. **Material data-quality errors.** The MI300 entry reports `computeUnits: 220`
   (MI300X is 304), `shaders: 14080` (19456), `memorySize: 128` (192 GB), and
   `fp32: 47.87 TFLOPS` (163 TFLOPS). No MI300X / gfx942 entry exists.
4. **Unit ambiguity.** Strix Halo entries report `computeUnits: 80` (Radeon
   8060S) against AMD's official "40 RDNA 3.5 CUs" for Ryzen AI Max+ 395 — a
   clean 2:1 ratio suggesting a non-standard (SIMD or dual-CU) counting basis.
5. **Provenance.** TechPowerUp is a third-party aggregator and does not satisfy
   the project's `opengpu` / `rocm-systems` upstream constraint for ISA values.

The only values it gets right (`wavefrontSize`, `ldsPerCU`, the "RDNA 3.5" /
"CDNA 3.0" labels) are already obtained directly from AMD official sources, so
the database adds no coverage. It is recorded here as "sampled and excluded"
rather than silently dropped.

## 11. Conclusions and Next Steps

- AMD official material fully covers the decision model: Omniperf supplies the
  taxonomy, the MI300X doc supplies the CDNA3 occupancy formula, GPUOpen
  supplies the RDNA occupancy model and the occupancy≠performance caveat, and
  the HIP Hardware Implementation chapter supplies the RDNA/CDNA divergence.
- The static path emits **Layer R resource signals only**, at `inferred_*`
  confidence, never a compute/memory-bound verdict.
- The data layer is decision-ready: the `ArchIsaBudget` divergence fields let
  the formula pick a tier, and `register_allocation_model` gates RDNA4 dynamic
  fallback.
- Open items for the `/gsd-plan-phase` workflow: (a) the `nW` gap (§9); (b) the
  `vgpr_limit` addressing-vs-physical semantics; (c) the `sol_execbench.decision.v1`
  schema and derivation run path; (d) cross-sidecar precedence against
  `profile_summary.v2` and `agent_feedback.v2`.

## References

### AMD official
- HIP Performance Guidelines — https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
- HIP Hardware Implementation (RDNA vs CDNA) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- MI300X Workload Optimization (occupancy formula) — https://rocm.docs.amd.com/en/docs-6.1.2/how-to/tuning-guides/mi300x/workload.html
- Omniperf Grafana Analysis (18-panel taxonomy, SOL thresholds) — https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.2.0/how-to/analyze/grafana-gui.html
- Occupancy Explained (GPUOpen; 4-way limiter, occupancy≠performance) — https://gpuopen.com/learn/occupancy-explained/
- Optimizing GPU Occupancy with Large Thread Groups (GPUOpen) — https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/
- CDNA 3 White Paper — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
- Using the Matrix Cores of RDNA 4 (GPUOpen; WMMA) — https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
- RDNA 4 @ Hot Chips 2025 (dynamic register allocation) — https://hc2025.hotchips.org/assets/program/conference/day1/8_amd_pomianowski_final.pdf
- AMD Ryzen AI Max+ 395 (gfx1150 = RDNA 3.5 confirmation) — https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-395-processor-breakthrough-ai-.html

### Research (AMD bottleneck modeling)
- Metrics and Design of an Instruction Roofline Model for AMD GPUs — https://www.researchgate.net/publication/359629923
- Extending the Roofline Model: Bottleneck Analysis (CMU/SPIRAL) — https://spiral.ece.cmu.edu/pub-spiral/pubfile/paper_181.pdf

### Third-party (assessed in §10)
- arXiv:2603.28793, Toward a Universal GPU ISA — https://arxiv.org/abs/2603.28793 (methodology reference; preprint, unrefereed, AMD unvalidated)
- RightNow-AI/RightNow-GPU-Database — https://github.com/RightNow-AI/RightNow-GPU-Database (sampled, excluded)
- Dynamic Register Allocation on RDNA 4 (Chips and Cheese) — https://chipsandcheese.com/p/dynamic-register-allocation-on-amds (corroborates Hot Chips 2025)

### Cross-vendor (validation only)
- NVIDIA Nsight Compute SOL — https://forums.developer.nvidia.com/t/what-is-sol-speed-of-light/191348
