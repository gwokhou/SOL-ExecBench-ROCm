# Final calibration review fixes

## Delivered

- Calibration artifacts now carry a canonical SHA-256 payload checksum.  Parsing
  rejects edited checksummed artifacts, and model building rejects legacy
  artifacts that lack the checksum.
- Calibration provenance records UUID, ROCm version, adapter policy, and
  pre/during/post clock observations.  Authority requires a real during-sample
  lock observation and an observed reset, rather than a successful lock command.
- `hardware-model build` rediscovers the selected live GPU and rejects UUID,
  architecture, ROCm-version, missing-evidence, and optional freshness mismatches.
- `--architecture` always follows runtime discovery and is now an assertion.
- Adapters declare BF16 matrix/MFMA candidates in addition to portable FP32;
  probe failures remain explicit unknown/unavailable evidence.  HIP benchmarks
  warm up before seven timed samples and reject non-finite measurements.
- Profiler discovery derives its requirements file and version from the actual
  installed launcher layout; it no longer assumes ROCm 7.1.1.
- Missing optional SOLAR derivation is represented as `not_requested`, not an
  artificial official-score blocker.

## Verification

```text
uv run pytest tests/sol_execbench/core/scoring/hardware_calibration \
  tests/sol_execbench/cli/commands/test_hardware_model_cli.py \
  tests/sol_execbench/core/scoring/test_amd_native_score.py \
  tests/sol_execbench/core/evidence/test_official_score_evidence.py -n 0 -q
# 74 passed

uv run --with ruff ruff check src/sol_execbench/core/scoring/hardware_calibration \
  src/sol_execbench/cli/commands/hardware_model.py \
  src/sol_execbench/core/scoring/amd_score/workload.py \
  src/sol_execbench/core/scoring/official_score.py \
  tests/sol_execbench/core/scoring/hardware_calibration \
  tests/sol_execbench/cli/commands/test_hardware_model_cli.py
# All checks passed

git diff --check
# clean
```

## Remaining concern

The new BF16 matrix candidates are declared and evidence-safe, but the current
portable HIP source remains FP32-only.  On a real system that cannot compile a
path-specific implementation it is recorded as `unknown`; it never produces a
false BF16 measurement.  A future ISA-specific kernel implementation is needed
before those BF16 profiles can become authoritative.

## Re-review follow-up

- The production HIP backend now explicitly marks non-FP32 or matrix paths as
  unavailable before compiling the FP32 source.  BF16 matrix candidates use the
  estimator-aligned `mfma` path and cannot become measured from that source.
- Every generated v3 profile evidence reference now embeds the input
  calibration checksum.
- Clock-policy reset runs even when lock acquisition reports failure or throws;
  post-reset observation is retained as evidence.
- Hardware-model parsing rejects NaN and either infinity for legacy scalar and
  v3 profile rates.

Re-review verification: the calibration, hardware-model, CLI, score, and
official-evidence focus ran with `-n 0`: **90 passed**.  Ruff passed for all
changed source and test paths.

## Final lifecycle follow-up

The entire clock-control/probe lifecycle is now enclosed by the reset
`finally` block.  Consequently, `--require-clock-lock` failures after a false
or exceptional lock result still call `unlock()` and attempt a post-reset
observation.  The strict false-lock regression verifies this behavior.  The
live RDNA4 smoke test intentionally accepts only the authority-safe rejected
diagnostic when strict evidence cannot be reached; it does not require all
matrix candidates to be measured or a successful authority upgrade.

Final focused verification: **91 passed** with `-n 0`; Ruff passed.
