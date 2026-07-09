---
status: complete
created: 2026-07-09
---

# Organize Runtime Evidence Modules

## Goal

Reduce flat `src/sol_execbench/core/evidence/` layout by moving the clear
`runtime_evidence*` feature cluster into a subpackage.

## Scope

- Move `runtime_evidence.py` to `runtime_evidence/__init__.py`.
- Move `runtime_evidence_builders.py` to `runtime_evidence/builders.py`.
- Move `runtime_evidence_collectors.py` to `runtime_evidence/collectors.py`.
- Move `runtime_evidence_io.py` to `runtime_evidence/io.py`.
- Move `runtime_evidence_models.py` to `runtime_evidence/models.py`.
- Move `runtime_evidence_cli.py` to `runtime_evidence/cli.py`.
- Add `runtime_evidence/__main__.py` for `python -m` execution.
- Update imports, scripts, docs, and tests.

## Out of Scope

- Other evidence modules (`baseline`, `baseline_export`, `scoring_guardrails`,
  shared checksum/ref helpers).
- Behavioral changes.
- Backward compatibility facade modules for old flat submodule paths.

## Verification

- Focused runtime evidence tests.
- Ruff check.
- Ty check.
- Full pytest if focused validation passes.
