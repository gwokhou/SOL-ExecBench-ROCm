---
phase: 41-bound-model-contract-and-hardware-artifacts
status: passed
verified: 2026-05-23
requirements: [HW-01, HW-02, HW-03, HW-04, DOC-01]
---

# Phase 41 Verification

## Verdict

Phase 41 passes automated verification.

## Goal

Establish the v2 bound artifact contract, hardware-model JSON loader, packaged
RDNA 4 default, and public-contract guardrails before downstream estimator and
artifact work.

## Requirement Coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| HW-01 | Passed | Versioned AMD hardware model JSON artifacts load through `load_amd_hardware_model()` and packaged `load_packaged_amd_hardware_model("gfx1200")`. |
| HW-02 | Passed | Invalid artifact fields, old `validation_status`, non-positive values, missing provenance, status errors, and architecture mismatches raise clear `ValueError`s. |
| HW-03 | Passed | Packaged defaults expose only RDNA 4 `gfx1200`; CDNA 3/CDNA 4 validated claims remain rejected/deferred. |
| HW-04 | Passed | Built-in/default models route through the same strict packaged JSON parser and carry split validation metadata. |
| DOC-01 | Passed | Guardrail tests keep canonical Trace JSONL, public schemas, and primary CLI help free of derived hardware/bound options. |

## Must-Haves Checked

- Packaged `gfx1200.json` exists in the wheel under `sol_execbench/data/amd_hardware_models/`.
- Packaged and external hardware model JSON share one strict parser.
- Hardware and model validation statuses are split; legacy `validation_status` is rejected.
- `default_amd_hardware_models()` resolves through packaged resources.
- Public contract guardrails prevent primary CLI/schema/Trace drift.

## Automated Checks

```text
uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_public_contract_guardrails.py -x
```

Result: 37 passed.

```text
uv build
```

Result: built `dist/sol_execbench-1.0.2.tar.gz` and `dist/sol_execbench-1.0.2-py3-none-any.whl`.

```text
uv run python -c "import zipfile; from pathlib import Path; wheels=list(Path('dist').glob('*.whl')); assert wheels; latest=max(wheels, key=lambda p: p.stat().st_mtime); names=zipfile.ZipFile(latest).namelist(); assert 'sol_execbench/data/amd_hardware_models/gfx1200.json' in names"
```

Result: passed.

## Human Verification

None required.

## Gaps

None.
