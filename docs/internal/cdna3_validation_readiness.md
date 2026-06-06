# CDNA 3 Validation Readiness

Phase 21 implemented readiness metadata for real CDNA 3 validation runs. Later
`gfx942` cloud runs recorded CDNA3 pytest and dataset-infrastructure evidence,
but they still do not constitute a completed benchmark-grade CDNA3 or MI300X
hardware-validation pass.

## Status

- Readiness implementation: available through internal diagnostics helpers.
- Pytest validation: recorded on real `gfx942` at repository HEAD `0d6c3e1`
  with `1401 passed, 62 skipped`.
- Dataset validation infrastructure: operational on real `gfx942`; full
  235-problem validation was run, with 220 complete passing problem traces, 15
  expected Quant NVFP4/MXFP4 CDNA3 skips, and known timeout blockers.
- Benchmark-grade hardware validation: blocked until the timeout shards,
  clock-lock evidence, timing evidence, AMD score evidence, FP8 status, and
  low-precision claim boundary are resolved or explicitly accepted.
- A real `gfx942` validation attempt on 2026-06-04 is recorded in
  [CDNA3 gfx942 Validation Attempt](cdna3_gfx942_validation_attempt.md). That
  initial attempt did not pass the full adapted suite. Follow-up fixes were
  re-run on `gfx942` at repository HEAD `0d6c3e1`, where the adapted pytest
  suite passed with `1401 passed, 62 skipped`.
- The pytest pass is CDNA3 `gfx942` adapted-suite validation evidence. It does
  not by itself complete MI300X benchmark-grade validation because dataset
  timeout blockers, clock-lock, timing, AMD score, FP8, and deferred
  low-precision status evidence are still required.
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
- Dataset summary and timeout-shard accounting for any benchmark dataset run.
- For MI300X (`gfx942`) validation, include clock-lock evidence, dataset
  summary, per-problem traces, ROCm timing evidence, AMD-native score report,
  FP8 status, NVFP4/MXFP4 deferred status, and expected result categories.

## Current Clock-Lock Blocker

On 2026-06-05, a cloud `gfx942` validation run attempted to use
`scripts/run_dataset.py --lock-clocks`. The benchmark correctly rejected the
run because the server still reported unlocked clocks:

```text
Clocks locked: no
lock_clocks=True but GPU clocks are not locked on this server
```

Manual `sudo rocm-smi --setperflevel manual` attempts did not transition the
device out of:

```text
Performance Level: auto
```

This indicates the cloud host or scheduler currently prevents effective DPM
clock locking from inside the validation session. Until the host can provide
locked-clock evidence, CDNA3 functional validation should run without
`--lock-clocks`, and timing output from that environment must be treated as
unlocked-clock evidence rather than benchmark-grade locked-clock timing.

## Acceptance Criteria

- Full adapted pytest suite completes successfully on a real CDNA 3 GPU.
- Recorded environment reports `gfx94*`.
- Dataset validation either completes without non-skip failures or records an
  explicit accepted-exclusion boundary for remaining timeout blockers.
- Support matrix claim is updated only after recorded evidence and claim
  boundaries exist.
- MI300X support status is not upgraded unless
  `mi300x_validation_claim_blockers()` returns no blockers.

## No-Claim Guardrail

Until the timeout, clock-lock, timing, score, and claim-boundary evidence exists,
documentation must say CDNA3/gfx942 validation infrastructure is operational
with known blockers, and must not say CDNA 3 hardware validation fully passed.
