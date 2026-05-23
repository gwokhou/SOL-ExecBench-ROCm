---
phase: 43-operator-flop-byte-movement-modeling
status: passed
verified: 2026-05-23
requirements: [MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05]
---

# Phase 43 Verification

## Verdict

Phase 43 passes automated verification.

## Goal

Implement auditable formulas and memory-movement evidence for core AMD
SOL/SOLAR operator families while preserving legacy v1 compatibility and
canonical public contracts.

## Requirement Coverage

| Requirement | Status | Evidence |
| --- | --- | --- |
| MODEL-01 | Passed | GEMM and batched GEMM rich estimates expose `gemm_flops`/`batched_gemm_flops`, formula strings, shape-derived formula inputs, FLOPs, confidence, and bytes. |
| MODEL-02 | Passed | Elementwise and activation estimates produce conservative per-node formulas, read/write byte evidence, and inexact confidence without fusion inference. |
| MODEL-03 | Passed | Reduction, normalization, and softmax estimates use axis metadata when available, mark missing axis evidence explicitly, and use conservative inexact pass-count formulas. |
| MODEL-04 | Passed | Data movement estimates distinguish logical views, broadcast views, materialized `contiguous`, and dtype conversion movement evidence. |
| MODEL-05 | Passed | `OperatorWorkEstimate` exposes per-node formula fields, read/write/intermediate/movement/total bytes, rationale, warnings, and legacy `WorkEstimate.bytes_accessed` adapts from rich `total_bytes`. |

## Must-Haves Checked

- `estimate_bound_work(BoundGraph)` is the primary rich estimator API.
- Every `BoundGraphNode` receives one rich operator estimate, including unsupported nodes.
- Out-of-scope families remain explicit unsupported estimates.
- Missing shape, dtype, axis, or tensor evidence downgrades confidence instead of fabricating supported work.
- Graph extractor metadata additions use `BoundGraphNode.attributes`; dataclass fields remain stable.
- Legacy `amd_sol.estimate_work()` remains a v1 adapter returning unchanged `WorkEstimate` fields.
- v1 AMD SOL bound artifacts do not serialize `formula_inputs`, `read_bytes`, `movement_bytes`, or `operator_work_estimates`.
- Canonical `Definition`, `Workload`, `Trace`, and primary CLI help remain unchanged.

## Automated Checks

```text
uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x
```

Result: 50 passed.

```text
uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py
```

Result: all checks passed.

## Human Verification

None required.

## Gaps

None.
