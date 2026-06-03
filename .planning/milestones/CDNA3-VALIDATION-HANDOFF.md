# CDNA 3 Validation Handoff

**Created:** 2026-05-21
**Milestone:** v1.1 CDNA 3 Support and Migration Closure
**Status:** Deferred to next milestone

## Purpose

v1.1 adds CDNA 3 code/schema support for `gfx94*` targets. It does not record a
real CDNA 3 hardware validation pass. The next milestone should validate the
adapted suite on at least one CDNA 3 GPU before documentation claims hardware
validation.

## Required Hardware

- AMD CDNA 3 GPU reporting a `gfx94*` architecture through PyTorch ROCm, such
  as `gfx940`, `gfx941`, or `gfx942`.
- ROCm >= 7.0.
- PyTorch ROCm environment matching the project lock/dependency configuration.

## Commands

Run the full adapted suite:

```bash
uv run --no-sync pytest tests/
```

Capture environment context:

```bash
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("name", torch.cuda.get_device_name(0))
    print("gcnArchName", getattr(props, "gcnArchName", ""))
PY
rocm-smi --showproductname --showdriverversion --showhw || true
rocminfo | grep -E "Name: *gfx94|Marketing Name" || true
```

## Evidence To Record

- Exact GPU name and `gfx94*` architecture.
- ROCm/HIP/PyTorch versions.
- Full pytest command and final pass/skip/fail counts.
- Any skipped tests and whether they are expected under CDNA 3.
- Any CDNA 3-specific deviations or follow-up issues.
- For MI300X (`gfx942`) validation, also record dataset summary, per-problem
  traces, ROCm timing evidence, clock-lock evidence, AMD-native score report,
  FP8 status, and NVFP4/MXFP4 `deferred_no_amd_path` status as defined in the
  MI300X validation handoff.

## Acceptance Criteria

- `uv run --no-sync pytest tests/` completes successfully on a real CDNA 3 GPU.
- The recorded environment shows `gfx94*`.
- Documentation support matrix can be updated from "code/schema support;
  hardware validation deferred" to "full adapted suite passed" only after this
  evidence exists.
- MI300X-specific hardware-validation claims require
  `mi300x_validation_claim_blockers()` to return no blockers.
