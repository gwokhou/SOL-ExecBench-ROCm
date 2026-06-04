# CDNA 3 Validation Readiness

Phase 21 implements readiness metadata for a future real CDNA 3 validation run.
It does not record a CDNA 3 hardware-validation pass.

## Status

- Readiness implementation: available through internal diagnostics helpers.
- Hardware validation: deferred until a full adapted suite run is recorded on
  real `gfx94*` hardware.
- A real `gfx942` validation attempt on 2026-06-04 is recorded in
  [CDNA3 gfx942 Validation Attempt](cdna3_gfx942_validation_attempt.md). That
  attempt did not pass the full adapted suite and does not upgrade CDNA3 or
  MI300X support claims.
- Claim wording: `cdna3_readiness_implemented` is not the same as CDNA 3
  hardware validated.
- Blocked or non-CDNA targets use `cdna3_hardware_validation_deferred`.

## Supported Target Detection

- CDNA 3 targets are `gfx94*`, including `gfx940`, `gfx941`, and `gfx942`.
- RDNA 4 targets such as `gfx1200` are not CDNA 3 validation targets.
- Unknown targets produce blockers instead of validation claims.

## Expected Commands

```bash
uv run --no-sync pytest tests/
uv run python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
rocm-smi --showproductname --showdriverversion --showhw || true
rocminfo | grep -E "Name: *gfx94|Marketing Name" || true
```

## Evidence Required

- Exact GPU name and `gfx94*` architecture.
- ROCm/HIP/PyTorch versions.
- Full pytest command and final pass/skip/fail counts.
- Expected skips and CDNA 3-specific deviations.
- For MI300X (`gfx942`) validation, include clock-lock evidence, dataset
  summary, per-problem traces, ROCm timing evidence, AMD-native score report,
  FP8 status, NVFP4/MXFP4 deferred status, and expected result categories.

## Acceptance Criteria

- Full adapted pytest suite completes successfully on a real CDNA 3 GPU.
- Recorded environment reports `gfx94*`.
- Support matrix claim is updated only after recorded evidence exists.
- MI300X support status is not upgraded unless
  `mi300x_validation_claim_blockers()` returns no blockers.

## No-Claim Guardrail

Until that evidence exists, documentation must keep using deferred/readiness
language and must not say CDNA 3 hardware validation passed.
