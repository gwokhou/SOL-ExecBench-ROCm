# MI300X Validation Readiness

This document is a readiness and status handoff for AMD Instinct MI300X
validation. Current MI308X (`gfx942`) cloud runs provide CDNA3 pytest and
dataset validation-infrastructure evidence, but they do not record a completed
MI300X commercial GPU hardware-validation pass.

## Scope

- Target hardware: AMD Instinct MI300X, CDNA 3, `gfx942`.
- Required ROCm baseline: ROCm >= 7.0 with PyTorch ROCm wheels.
- Clock policy: clocks must be locked before benchmark-grade measurements.
- FP8: validate on MI300X once hardware access exists.
- NVFP4/MXFP4: deferred; no AMD hardware-validation path is claimed in this
  project yet.

## Current Evidence Status

- CDNA3 pytest validation passed on real MI308X (`gfx942`) at repository HEAD
  `0d6c3e1` with `1401 passed, 62 skipped`.
- Full 235-problem dataset validation was run on MI308X (`gfx942`). The
  corrected interpretation is 220 complete passing problem traces, 15 expected
  Quant NVFP4/MXFP4 CDNA3 skips, and 6 remaining timeout shards across 4
  problems.
- Nested `src/sol_execbench/driver/templates/eval_driver.py` timeout classification was fixed in commit `2984c29`
  and verified on
  `FlashInfer-Bench/014_gqa_paged_prefill_causal_h32_kv4_d128_ps1`, which now
  records `29 PASSED + 1 TIMEOUT` and summary `FAIL`.
- Cloud clock locking remains blocked in the tested environment: the server
  reported `Performance Level: auto` after manual `rocm-smi` attempts. Timing
  from that environment is unlocked-clock evidence, not benchmark-grade
  locked-clock timing.
- Quant NVFP4/MXFP4 skips on CDNA3 are expected, not correctness failures.
  Hardware validation for those formats requires CDNA4-class support.
- MI308X and MI300X share the `gfx942` code path, but this evidence is not a
  completed MI300X validation pass because the hardware configurations differ.

Remaining MI300X claim blockers:

- Resolve or explicitly accept the 6 dataset timeout shards on exact MI300X
  hardware.
- Record locked-clock evidence or keep timing claims bounded as unlocked-clock
  evidence.
- Archive timing evidence, AMD-native score report, FP8 status, and expected
  result categories for the accepted validation scope.
- Keep NVFP4/MXFP4 status as `deferred_no_amd_path` on CDNA3.

## Commands

```bash
rocminfo | grep -E "Name: *gfx94|Marketing Name" || true
rocm-smi --showproductname --showdriverversion --showhw || true
uv run python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
uv run pytest tests/
uv run scripts/run_dataset.py <path-to-SOL-ExecBench-benchmark> \
  --output out/mi300x-validation \
  --lock-clocks \
  --timing-evidence-dir out/mi300x-timing \
  --gpu-architecture gfx942 \
  --amd-score-report out/mi300x-amd-score.json \
  --scoring-baseline <path-to-mi300x-scoring-baseline.json>
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

## Expected Result Categories

The validation archive must classify and record:

- expected skips
- missing tools
- functional failures
- timing instability
- missing evidence
- FP8 validation
- deferred quantization formats

## Acceptance Criteria

- Full adapted pytest suite passes on real MI300X hardware.
- Dataset validation run completes with expected skips/deviations documented and
  no unaccepted non-skip failures.
- Environment evidence records MI300X and `gfx942`.
- Clock locking is enabled and recorded.
- Per-problem traces, ROCm timing evidence, and AMD-native score report are
  archived.
- Expected result categories are present even when empty.
- Reports are not marked MI300X hardware-validated under CDNA3 unless
  `mi300x_validation_claim_blockers()` returns no blockers.

## No-Claim Rule

Until the evidence above exists, public docs and reports must say current
CDNA3/gfx942 validation infrastructure evidence was recorded on MI308X with
known blockers, not that MI300X has a completed benchmark-grade
hardware-validation claim.

Readiness metadata is not a validation claim.
