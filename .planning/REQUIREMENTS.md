# Requirements: SOL ExecBench ROCm Port

**Defined:** 2026-06-04
**Milestone:** v1.28 CDNA3 Test and Documentation Readiness
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.28 Requirements

### CDNA3 Test Surface

- [ ] **CDNA3-TEST-01**: Maintainer can run at least one
  `@pytest.mark.requires_cdna3` test path that executes only on real `gfx94*`
  ROCm hardware and skips with a clear reason elsewhere.
- [ ] **CDNA3-TEST-02**: Maintainer can verify CDNA3-specific architecture
  detection, marker selection, and skip behavior without requiring current
  machine access to MI300X.
- [ ] **CDNA3-TEST-03**: Maintainer can exercise CDNA3 native-build or
  packaging paths for `gfx940`, `gfx941`, and `gfx942` metadata without
  confusing schema/build support with hardware validation.
- [ ] **CDNA3-TEST-04**: Maintainer can select CDNA3 hardware tests through
  documented pytest commands that do not accidentally include RDNA 4-only or
  legacy NVIDIA-only validation paths.

### MI300X Evidence Contract

- [ ] **MI300X-EVID-01**: Future validator can follow a documented MI300X
  validation command sequence covering environment capture, full pytest suite,
  dataset run, timing evidence, clock-lock evidence, traces, and score reports.
- [ ] **MI300X-EVID-02**: Future validator can determine which artifacts are
  required before any report or support matrix may mark CDNA3/MI300X as
  hardware-validated.
- [ ] **MI300X-EVID-03**: Future validator can distinguish expected skips,
  tool-unavailable states, functional failures, timing instability, and missing
  evidence in the MI300X validation handoff.
- [ ] **MI300X-EVID-04**: Future validator can include FP8 validation results
  for MI300X when FP8 workloads are present while keeping NVFP4/MXFP4
  validation deferred.

### Deferred Validation Guardrails

- [ ] **CDNA3-GATE-01**: CPU-safe tests prove CDNA3 schema/build/readiness
  support cannot be upgraded into a CDNA3 hardware-validation claim without a
  complete real-hardware evidence chain.
- [ ] **CDNA3-GATE-02**: Public docs, readiness reports, score warnings, and
  release materials continue to state that current CDNA3/MI300X full-suite
  validation is deferred until real `gfx94*` evidence exists.
- [ ] **CDNA3-GATE-03**: Claim-upgrade or validation-readiness helpers expose
  blockers when MI300X identity, `gfx942` architecture, full-suite result,
  clock evidence, trace artifacts, or score reports are missing.
- [ ] **CDNA3-GATE-04**: Test-suite audit coverage fails if `requires_cdna3`
  remains only a registered marker with no concrete hardware-gated test path.

### Public Documentation

- [ ] **CDNA3-DOC-01**: Testing documentation explains how to run CDNA3-only
  tests, how they skip on non-CDNA3 machines, and why skipped CDNA3 tests do not
  imply validation failure on the current host.
- [ ] **CDNA3-DOC-02**: ROCm support docs distinguish CDNA3 schema support,
  CDNA3 test readiness, deferred MI300X full-suite validation, and unavailable
  CDNA4 validation.
- [ ] **CDNA3-DOC-03**: Internal handoff docs identify the exact future
  milestone gate for changing CDNA3 support status from readiness to
  hardware-validated.
- [ ] **CDNA3-DOC-04**: Contributor-facing docs describe where to add future
  CDNA3 tests and which markers/evidence requirements apply.

## Future Requirements

Deferred to a later milestone because they require hardware access, broader
research scope, or operational infrastructure outside v1.28.

### Hardware Validation

- **HWVAL-01**: Validator can run and archive the full adapted pytest suite on
  real AMD Instinct MI300X (`gfx942`) hardware.
- **HWVAL-02**: Validator can run a bounded or full dataset validation on
  MI300X with clock-lock evidence, timing evidence, traces, and AMD-native
  score artifacts.
- **HWVAL-03**: Validator can upgrade the support matrix from CDNA3 readiness
  to CDNA3 hardware validation only after all required MI300X artifacts exist.

### Broader Validation

- **PAPER-01**: Validator can perform full 235-problem paper-scale validation
  and compare results against the paper-aligned denominator.
- **CDNA4-01**: Validator can define and execute validation on future CDNA4
  hardware once suitable devices are accessible.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Actual CDNA3/MI300X full-suite execution in v1.28 | The current machine cannot execute a real `gfx94*` hardware validation pass. |
| Claiming CDNA3 hardware validation | Requires complete real-hardware evidence that this milestone explicitly defers. |
| CDNA4 validation | Suitable CDNA4 hardware is not currently accessible. |
| Full 235-problem paper-scale validation | Separate large validation effort; not needed to make CDNA3 tests/docs ready. |
| Upstream SOLAR parity or hosted leaderboard authority | Separate research/operations scope outside CDNA3 readiness. |
| Hard multi-tenant sandboxing | Security architecture is still deferred and unrelated to CDNA3 marker/docs readiness. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CDNA3-TEST-01 | Phase 127 | Pending |
| CDNA3-TEST-02 | Phase 127 | Pending |
| CDNA3-TEST-03 | Phase 127 | Pending |
| CDNA3-TEST-04 | Phase 127 | Pending |
| MI300X-EVID-01 | Phase 128 | Pending |
| MI300X-EVID-02 | Phase 128 | Pending |
| MI300X-EVID-03 | Phase 128 | Pending |
| MI300X-EVID-04 | Phase 128 | Pending |
| CDNA3-GATE-01 | Phase 129 | Pending |
| CDNA3-GATE-02 | Phase 129 | Pending |
| CDNA3-GATE-03 | Phase 129 | Pending |
| CDNA3-GATE-04 | Phase 129 | Pending |
| CDNA3-DOC-01 | Phase 130 | Pending |
| CDNA3-DOC-02 | Phase 130 | Pending |
| CDNA3-DOC-03 | Phase 130 | Pending |
| CDNA3-DOC-04 | Phase 130 | Pending |

**Coverage:**
- v1.28 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-06-04*
*Last updated: 2026-06-04 after v1.28 milestone start*
