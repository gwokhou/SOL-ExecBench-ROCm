# Requirements: SOL ExecBench ROCm Port v1.15

**Defined:** 2026-05-25
**Milestone:** v1.15 Research-Grade ROCm Benchmark Release
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL ExecBench.

## v1.15 Requirements

### Claims And Positioning

- [x] **CLAIM-01**: The project documents which current results can be claimed as ROCm-port evidence, AMD-native-derived evidence, or release-preview evidence.
- [x] **CLAIM-02**: The project documents which claims remain explicitly unsupported, including NVIDIA B200 parity, official leaderboard parity, full 235-problem validation, upstream SOLAR parity, and unvalidated CDNA 3/CDNA 4 hardware claims.
- [x] **CLAIM-03**: Claim documentation maps each allowed claim to required evidence artifacts, tests, and commands.
- [x] **CLAIM-04**: Public docs and guardrail tests prevent curated-slice or AMD-native-derived results from being described as paper-level parity.

### Curated ROCm Benchmark Slice

- [x] **SLICE-01**: The project defines a curated ROCm benchmark slice with representative problems across PyTorch ROCm, Triton ROCm, HIP/C++, and at least one ROCm native library path when available.
- [x] **SLICE-02**: The curated slice records deterministic selection criteria, problem list, workload scope, hardware assumptions, and excluded categories.
- [x] **SLICE-03**: The curated slice can be executed through existing `sol-execbench` or dataset-runner paths without introducing a second benchmark runner.
- [x] **SLICE-04**: Curated-slice execution records canonical traces plus expected sidecar artifacts or explicit unscored/unavailable reasons.

### Researcher Workflows

- [x] **RESEARCH-01**: `docs/RESEARCHER-GUIDE.md` explains entry points for GPU kernel authors, compiler/backend researchers, agent kernel-optimization researchers, and benchmark/reproducibility researchers.
- [x] **RESEARCH-02**: The researcher guide explains how to interpret canonical traces, environment evidence, profiling evidence, AMD-native score reports, and readiness/closure reports.
- [x] **RESEARCH-03**: The project documents how to add or adapt a kernel solution without violating solution schema, reward-hack, or claim-boundary constraints.
- [x] **RESEARCH-04**: The project documents how agent-optimizer experiments should use the harness, traces, and guardrails without treating derived artifacts as hidden scoring authority.

### Cookbooks And Reproducibility

- [x] **COOK-01**: The project provides end-to-end cookbook recipes for single-kernel evaluation, adding a HIP/Triton solution, running the curated slice, generating AMD-native score evidence, and collecting `rocprofv3` diagnostics.
- [x] **COOK-02**: Cookbook commands are copy-pasteable and use stable repository paths or explain required dataset/hardware prerequisites.
- [x] **REPRO-01**: The milestone produces a release closure document that lists curated-slice commands, generated artifact families, expected pass/fail/skip semantics, and known gaps.
- [x] **REPRO-02**: Release closure distinguishes local reproducibility evidence from paper parity, leaderboard readiness, and full hardware validation.

## Future Requirements

### Paper Parity

- **PAPER-01**: Recreate or verify the original paper-scale 124-model / 235-problem extraction and curation pipeline.
- **PAPER-02**: Run and report full 235-problem ROCm validation with complete denominator accounting.
- **PAPER-03**: Compare local SOLAR derivation behavior against upstream SOLAR at paper scale.

### Hardware And Static Evidence

- **HW-01**: Complete CDNA 3 / MI300X full-suite validation with archived environment, clock, trace, timing, and score evidence.
- **HW-02**: Add CDNA 4 validation when suitable hardware and ROCm support are available.
- **STATIC-01**: Add static kernel evidence with RGA/code-object analysis and GPUOpen ISA classification.

### Service Surface

- **SERV-01**: Add hosted leaderboard or submission service support after local claims, evidence, and curated release workflows are stable.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full 235-problem paper parity | Too broad for this milestone; v1.15 focuses on a small complete research preview. |
| Original 124-model extraction and curation | Requires a separate paper-parity milestone and substantial dataset work. |
| Upstream SOLAR equivalence claim | Current SOLAR evidence is AMD-local and derived; equivalence requires dedicated comparison. |
| CDNA 3 / MI300X validation claim | Requires real `gfx94*` full-suite execution evidence not available in this milestone scope. |
| Hosted leaderboard | Premature until the local reproducibility and claim model are stable. |
| Static RGA/code-object analysis | Valuable next milestone candidate, but intentionally deferred from v1.15. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLAIM-01 | Phase 64 | Complete |
| CLAIM-02 | Phase 64 | Complete |
| CLAIM-03 | Phase 64 | Complete |
| CLAIM-04 | Phase 64 | Complete |
| SLICE-01 | Phase 65 | Complete |
| SLICE-02 | Phase 65 | Complete |
| SLICE-03 | Phase 65 | Complete |
| SLICE-04 | Phase 65 | Complete |
| RESEARCH-01 | Phase 66 | Complete |
| RESEARCH-02 | Phase 66 | Complete |
| RESEARCH-03 | Phase 66 | Complete |
| RESEARCH-04 | Phase 66 | Complete |
| COOK-01 | Phase 66 | Complete |
| COOK-02 | Phase 66 | Complete |
| REPRO-01 | Phase 67 | Complete |
| REPRO-02 | Phase 67 | Complete |

**Coverage:**
- v1.15 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-05-25*
*Last updated: 2026-05-25 after v1.15 autonomous implementation*
