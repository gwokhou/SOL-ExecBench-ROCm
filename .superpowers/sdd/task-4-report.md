# Task 4 report: official authority gate and CLI

Implemented serialized `BoundEligibilityEvidence` for AMD-native scores and
made legacy scores without it official-blocked.  The official gate now rejects
non-scored AMD SOL/SOLAR aggregates, non-measured profiles, non-validated
hardware/model states, and degraded/inexact/unsupported bound warnings.

Added lazy `hardware-model` CLI dispatch with calibration and build commands.
Failures write a diagnostic rejected JSON artifact before Click exits nonzero.
Calibration remains diagnostic; building an external v3 model requires a fully
validated calibration input.

Verification:

- `uv run pytest tests/sol_execbench/core/scoring/hardware_calibration tests/sol_execbench/core/scoring/test_amd_hardware_models.py tests/sol_execbench/core/scoring/test_amd_sol_v2.py tests/sol_execbench/core/scoring/test_amd_native_score.py tests/sol_execbench/core/evidence/test_official_score_evidence.py tests/sol_execbench/cli/commands/test_hardware_model_cli.py -n 0 -q` — 86 passed.
- `uv run --with ruff ruff check .` — passed.
- The requested `test_live_calibration.py` path is absent in this checkout, so the marker-gated live check could not be collected.
