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

## Review follow-up

Added the missing marker-gated live calibration test at
`tests/sol_execbench/core/scoring/hardware_calibration/test_live_calibration.py`.
It invokes the offline CLI flow, writes beneath `tmp_path / "out"`, and verifies
the RDNA4 architecture, validated measurement candidates, and explicit unknown
profiler state.  Provisional calibration now writes the rejected diagnostic JSON
envelope before Click exits nonzero.  The profiler builder records `unknown` when
the managed profiler environment is unavailable, rather than omitting the state.

Verification: the exact `-m requires_rdna4` live test passed on this host, and
the CLI regression plus calibration builder tests passed (9 tests total).
