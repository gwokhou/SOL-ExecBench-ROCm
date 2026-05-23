---
phase: 42-structured-bound-graph-ir
status: passed
verified: 2026-05-23
requirements: [IR-01, IR-02, IR-03, IR-04]
---

# Phase 42 Verification

## Verdict

Phase 42 passes automated verification.

## Goal

Introduce a stable bound graph/IR that downstream formula and artifact code can
consume without depending on raw AST details.

## Requirement Coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| IR-01 | Passed | `build_bound_graph(definition, workload)` builds a derived `BoundGraph` without changing canonical schemas. |
| IR-02 | Passed | `BoundGraphNode`, `BoundTensor`, and `BoundEdge` expose stable IDs, op families, source expressions, tensor roles, shapes, dtypes, confidence, and rationale. |
| IR-03 | Passed | Unsupported calls, dynamic control flow, and trace failures are preserved as unsupported/inexact nodes and warnings. |
| IR-04 | Passed | `amd_sol.extract_graph()`, `estimate_work()`, and `build_amd_sol_bound_artifact()` continue to pass compatibility tests through the facade. |

## Must-Haves Checked

- Paper-aligned graph IR exists in `src/sol_execbench/core/scoring/amd_bound_graph.py`.
- Dynamic `torch.fx` extraction is attempted before AST fallback.
- AST fallback preserves unsupported/inexact graph evidence.
- Legacy AMD SOL artifact v1 behavior remains compatible.
- Public scoring exports are deliberate and covered by tests.
- Canonical `Definition`, `Workload`, `Trace`, and primary CLI help remain unchanged.

## Automated Checks

```text
uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x
```

Result: 46 passed.

```text
uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py
```

Result: all checks passed.

## Human Verification

None required.

## Gaps

None.

