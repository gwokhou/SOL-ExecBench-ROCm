# MI300X Validation Readiness

This document is a readiness handoff for a future AMD Instinct MI300X validation
run. It does not record a commercial GPU hardware-validation pass.

## Scope

- Target hardware: AMD Instinct MI300X, CDNA 3, `gfx942`.
- Required ROCm baseline: ROCm >= 7.0 with PyTorch ROCm wheels.
- Clock policy: clocks must be locked before benchmark-grade measurements.
- FP8: validate on MI300X once hardware access exists.
- NVFP4/MXFP4: deferred; no AMD hardware-validation path is claimed in this
  project yet.

## Commands

```bash
rocminfo | grep -E "Name: *gfx94|Marketing Name" || true
rocm-smi --showproductname --showdriverversion --showhw || true
uv run python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
uv run pytest tests/
uv run scripts/run_dataset.py data/SOL-ExecBench/benchmark \
  --output out/mi300x-validation \
  --lock-clocks \
  --timing-evidence-dir out/mi300x-timing \
  --gpu-architecture gfx942 \
  --amd-score-report out/mi300x-amd-score.json \
  --scoring-baseline baselines/mi300x-scoring-baseline.json
```

## Evidence To Record

- Exact GPU name: must identify AMD Instinct MI300X.
- Exact gfx target: must be `gfx94*`, expected `gfx942`.
- ROCm/HIP/PyTorch versions.
- Clock-lock evidence from `rocm-smi` and benchmark config.
- Full adapted pytest log and final pass/skip/fail counts.
- Dataset summary, per-problem traces, timing evidence, and AMD-native score
  report.
- FP8 validation result, or `deferred_no_case` if no FP8 workload is included in
  the selected validation slice.
- NVFP4/MXFP4 status: must remain `deferred_no_amd_path`.

## Acceptance Criteria

- Full adapted pytest suite passes on real MI300X hardware.
- Dataset validation run completes with expected skips/deviations documented.
- Environment evidence records MI300X and `gfx942`.
- Clock locking is enabled and recorded.
- Reports are not marked MI300X/CDNA3 hardware-validated unless
  `mi300x_validation_claim_blockers()` returns no blockers.

## No-Claim Rule

Until the evidence above exists, public docs and reports must say MI300X/CDNA3
hardware validation is deferred. Readiness metadata is not a validation claim.
