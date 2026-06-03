# MI300X Validation Handoff

**Status:** ready for future hardware execution; no MI300X validation pass is
recorded.

## Required Hardware And Stack

- AMD Instinct MI300X (`gfx942`, CDNA 3).
- ROCm >= 7.0.
- PyTorch ROCm environment with visible GPU.
- `hipcc`, `rocminfo`, `rocm-smi`, and `rocprofv3` on `PATH`.
- GPU clocks locked before benchmark-grade timing.

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

- `pytest_full_suite_log`
- `run_dataset_summary`
- `environment_report`
- `clock_lock_evidence`
- `per_problem_traces`
- `rocm_timing_evidence`
- `amd_native_score_report`
- `fp8_validation_result`: FP8 validation result for MI300X when an FP8 workload
  is included.
- `nvfp4_mxfp4_deferred_status`

The archive should include per-problem traces, ROCm timing evidence, and an
AMD-native score report with stable relative paths from the run summary.

## Expected Result Categories

Record all of these categories explicitly, even when the category has no
entries:

- `expected_skips`: tests or problems skipped for documented CDNA3 reasons.
- `missing_tools`: missing tools from the ROCm stack such as `rocminfo`, `rocm-smi`,
  `hipcc`, or `rocprofv3`.
- `functional_failures`: correctness, compile, runtime, or dataset failures.
- `timing_instability`: timing instability from clock-lock, repeatability, or
  timing-quality issues.
- `missing_evidence`: absent traces, timing evidence, logs, score reports, or
  environment sidecars.
- `fp8_validation`: FP8 result when workloads exist, otherwise
  `deferred_no_case`.
- `deferred_quantization_formats`: NVFP4/MXFP4 remains deferred.

## Deferred

- NVFP4/MXFP4 validation remains `deferred_no_amd_path`.

## Acceptance Criteria

- Full adapted suite passes on MI300X.
- Environment records AMD Instinct MI300X and `gfx942`.
- Clock locking is active and recorded.
- Dataset summary, per-problem traces, ROCm timing evidence, and AMD-native
  score report are archived.
- Expected result categories are recorded.
- `mi300x_validation_claim_blockers()` returns no blockers before any report or
  support matrix marks MI300X as hardware-validated on CDNA3.
