# Feature Research

**Domain:** AMD-native SOL scoring and ROCm profiler timing
**Researched:** 2026-05-22
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| AMD hardware model registry | SOL bounds need peak compute, bandwidth, clock policy, and architecture identity. | MEDIUM | Start with explicit YAML/JSON/Pydantic records for `gfx1200` and extensible `gfx94*` placeholders without validation claims. |
| SOLAR-like graph extraction path | User selected a SOLAR-like pipeline rather than a config-only roofline. | HIGH | Must handle PyTorch references, custom inputs, dynamic axes, and unsupported ops with auditable fallback states. |
| FLOP/byte analysis engine | SOL bounds require compute and memory traffic estimates. | HIGH | Begin with known op analyzers and clear unsupported-op diagnostics; do not fabricate precision for unsupported kernels. |
| AMD SOL-bound artifact | Scores need a concrete bound input per workload. | MEDIUM | Emit separate analysis artifacts so trace JSONL remains stable unless an additive output is approved. |
| AMD-native scoring workflow | Existing `sol_score()` is only a formula helper. | MEDIUM | Add workload and suite aggregation, baseline ingestion, score output, and claim-level guardrails. |
| Profiler-backed default timing | User selected replacing default event timing. | HIGH | Must preserve correctness loop and reward-hack defenses while invoking profiler around measured work only. |
| Source-specific timing semantics | User requires Triton/HIP/PyTorch timing口径 review. | HIGH | Expose `source_type`, `timer_backend`, `aggregation_rule`, and limitations in reports/artifacts. |

### Differentiators (Competitive Advantage)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Accuracy-first chimney model | Avoids false precision from a single timing口径 that cannot fit all operator sources. | MEDIUM | This is the key user-requested policy: split by source type when needed. |
| Bound confidence levels | Makes SOL-like results useful before every op analyzer is perfect. | MEDIUM | Mark bounds as exact/formula/manual/unsupported and prevent unsupported bounds from silently scoring. |
| Profiler evidence bundle | Makes timing claims auditable and reviewable. | MEDIUM | Store raw profiler trace path, parsed kernel rows, aggregation summary, tool versions, and clock policy. |
| PyTorch op attribution cross-check | Helps explain PyTorch references whose device work spans many library kernels. | HIGH | Use PyTorch profiler or annotations for attribution, but keep kernel trace as timing authority when possible. |
| Timing drift comparison | Provides confidence when replacing event timing. | MEDIUM | Compare profiler timing to existing event timing on focused fixtures before making profiler default. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| One score without bound evidence | Simple reports look cleaner. | It hides unsupported SOL bounds and can imply false AMD hardware validity. | Require bound artifact and claim level for every score. |
| One unified timer for every source type | Easier API and tables. | It may be inaccurate for Triton JIT/autotune, HIP native kernels, or PyTorch library dispatch. | Source-specific timer semantics with explicit backend mapping. |
| Replacing trace JSONL fields in place | Direct output looks convenient. | It breaks established public trace consumers. | Add sidecar scoring/timing reports first; only add trace fields via documented additive contract. |
| CDNA 3 validation upgrade | It closes a visible deferred item. | User excluded real CDNA 3 validation from this milestone. | Keep CDNA 3 as code/schema/readiness only. |

## Feature Dependencies

```text
AMD hardware model registry
    -> FLOP/byte analysis engine
        -> AMD SOL-bound artifact
            -> AMD-native scoring workflow

Profiler-backed default timing
    -> Source-specific timing semantics
        -> Profiler evidence bundle
            -> Timing drift comparison

Graph extraction path
    -> FLOP/byte analysis engine
```

### Dependency Notes

- **Hardware model before scoring:** SOL Score needs `T_SOL`; an AMD score cannot be valid without a target hardware model and clock policy.
- **Graph extraction before broad FLOP/byte coverage:** The analyzer needs a stable intermediate representation before op formulas scale.
- **Timing semantics before default replacement:** Replacing the default timer must first establish what is measured for Triton, HIP native, and PyTorch sources.
- **Evidence before claims:** Profiler output and bound artifacts should be stored before reports claim AMD-native score validity.

## MVP Definition

### Launch With (v1.5)

- [ ] AMD hardware model schema and first RDNA 4 model entry with explicit confidence/claim level.
- [ ] SOLAR-like graph extraction prototype for a focused subset of existing reference problems.
- [ ] FLOP/byte analyzers for a narrow set of high-value ops, plus unsupported-op diagnostics.
- [ ] AMD SOL-bound artifact and parser.
- [ ] AMD-native scoring command or workflow that consumes traces, baselines, and bound artifacts.
- [ ] ROCm profiler-backed timing backend selected as default through the benchmark config.
- [ ] Source-specific timing semantics for `triton`, `hip_native`, and `pytorch`.
- [ ] Tests and docs proving CDNA 3 validation remains deferred.

### Add After Validation (v1.x)

- [ ] Broader op analyzer coverage for attention, MoE, normalization, and backward patterns.
- [ ] Richer profiler correlation with ROCTx markers and PyTorch operator attribution.
- [ ] Repeated-sample statistical score/timing confidence intervals.
- [ ] Optional rocpd/SQLite parser when CSV is insufficient.

### Future Consideration (v2+)

- [ ] Full AMD SOLAR-equivalent support across the public 235-problem dataset.
- [ ] Public AMD leaderboard policy.
- [ ] Real CDNA 3 full-suite validation and support-matrix upgrade.
- [ ] Hardware counter-based roofline calibration beyond static peak specs.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Source-specific timing semantics | HIGH | MEDIUM | P1 |
| Profiler-backed default timing | HIGH | HIGH | P1 |
| AMD hardware model registry | HIGH | MEDIUM | P1 |
| SOL-bound artifact format | HIGH | MEDIUM | P1 |
| SOLAR-like graph extraction prototype | HIGH | HIGH | P1 |
| Initial FLOP/byte analyzers | HIGH | HIGH | P1 |
| AMD-native scoring workflow | HIGH | MEDIUM | P1 |
| PyTorch op attribution cross-check | MEDIUM | HIGH | P2 |
| rocpd parser | MEDIUM | MEDIUM | P2 |
| Full 235-problem coverage | HIGH | VERY HIGH | P3 |

## Sources

- arXiv 2603.19173 - SOL ExecBench benchmark design, SOLAR, SOL Score, and robust evaluation harness.
- ROCprofiler-SDK `rocprofv3` documentation - kernel trace, HIP runtime trace, HSA trace, output controls.
- PyTorch profiler documentation - operator/device activity profiling and activity groups.
- PyTorch HIP semantics documentation - ROCm reuse of `torch.cuda` interfaces.
- ROCm Triton optimization documentation - Triton AMD kernel optimization context.

---
*Feature research for: AMD-native SOL scoring and ROCm profiler timing*
*Researched: 2026-05-22*
