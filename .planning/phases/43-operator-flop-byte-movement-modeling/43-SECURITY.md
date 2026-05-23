---
phase: 43
slug: operator-flop-byte-movement-modeling
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-23
verified: 2026-05-23
---

# Phase 43 — Security

Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| BoundGraph -> rich estimates | Derived graph metadata becomes scored evidence. | Operation families, tensor IDs, shapes, dtypes, attributes, confidence, rationale. |
| Rich estimates -> public API | New formula and byte evidence must stay in derived scoring APIs unless intentionally adapted. | Formula fields, byte buckets, warning evidence. |
| Tensor metadata -> formulas | Shape and dtype metadata determine FLOP and byte evidence. | `BoundTensor.shape`, `BoundTensor.dtype`, node-local tensor references. |
| Graph sequence -> chain evidence | Multiple graph nodes can represent a chain but must not be silently fused. | Per-node operation order and tensor edges. |
| Source expressions -> attributes | Literal axis, dtype, and movement hints are extracted from AST or FX reference evidence. | `dim`/axis, `target_dtype`, `movement_kind`. |
| Movement evidence -> bytes | Logical views, broadcast views, materialization, and dtype conversion must not be conflated. | Read/write/movement byte estimates and movement rationale. |
| Rich estimates -> v1 WorkEstimate | Rich evidence is degraded into the legacy v1 compatibility shape. | FLOPs, total bytes, confidence, rationale. |
| Derived scoring APIs -> canonical contracts | Rich fields must not appear in primary CLI, schemas, or Trace JSONL. | Public schema dumps, CLI help, v1 artifact payloads. |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-43-01 | Tampering | derived estimate serialization | mitigate | `OperatorWorkEstimate` is a frozen dataclass with JSON-safe `to_dict()` enum/string serialization; covered by `test_operator_work_estimate_is_frozen_and_serializes_json_like_values`. | closed |
| T-43-02 | Repudiation | unsupported operation evidence | mitigate | `estimate_bound_work()` emits one explicit zero-valued unsupported estimate per unsupported/out-of-scope node; covered by unsupported and out-of-scope family tests. | closed |
| T-43-SC | Tampering | package dependencies | accept | Phase 43 added no package dependencies; dependency risk accepted as no dependency-surface change. | closed |
| T-43-03 | Tampering | formula inputs | mitigate | GEMM/BMM formulas derive `M`, `N`, `K`, and `B` from concrete `BoundTensor.shape`; exact formula input tests cover matmul and batched matmul. | closed |
| T-43-04 | Repudiation | metadata gaps | mitigate | Missing shape/dtype zeros affected buckets, downgrades confidence, and records missing evidence in rationale; covered by metadata gap tests. | closed |
| T-43-05 | Tampering | axis metadata | mitigate | Axis metadata is evidence only; missing axis produces `axis_source == "missing"` and inexact all-elements estimates; covered by reduction/softmax tests. | closed |
| T-43-06 | Tampering | movement bytes | mitigate | Logical/broadcast views produce zero movement bytes and materialized movement produces nonzero movement bytes; covered by view, broadcast, contiguous, and dtype conversion tests. | closed |
| T-43-07 | Tampering | v1 artifact schema | mitigate | `AMD_SOL_SCHEMA_VERSION` and legacy `WorkEstimate` fields are preserved; v1 artifact tests assert schema version and dataclass fields. | closed |
| T-43-08 | Information Disclosure | derived fields in canonical outputs | mitigate | Public-contract tests reject `formula_kind`, `read_bytes`, `movement_bytes`, `operator_work_estimates`, and new primary CLI options in canonical outputs. | closed |
| T-43-09 | Repudiation | unsupported compatibility estimates | mitigate | Unsupported rich estimates adapt to unsupported `WorkEstimate` and unsupported `OpSolBound` instead of disappearing; covered by v1 unsupported artifact tests. | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-43-01 | T-43-SC | Phase 43 introduced no new dependencies, so package dependency tampering risk is accepted as unchanged dependency surface. | GSD security audit | 2026-05-23 |

---

## Audit Evidence

| Area | Evidence |
|------|----------|
| Rich estimate immutability/serialization | `src/sol_execbench/core/scoring/amd_bound_estimates.py`; `tests/sol_execbench/test_amd_bound_estimates.py` |
| Unsupported and metadata-gap behavior | `tests/sol_execbench/test_amd_bound_estimates.py`; `tests/sol_execbench/test_amd_bound_graph.py` |
| Axis and movement extraction | `src/sol_execbench/core/scoring/amd_bound_graph.py`; `src/sol_execbench/core/scoring/amd_bound_estimates.py` |
| Legacy v1 adapter and schema preservation | `src/sol_execbench/core/scoring/amd_sol.py`; `tests/sol_execbench/test_amd_sol_bounds.py` |
| Public contract leakage guardrails | `tests/sol_execbench/test_public_contract_guardrails.py` |

## Automated Checks Reviewed

```text
uv run pytest tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x
```

Result recorded in `43-VERIFICATION.md`: 50 passed.

```text
uv run --with ruff ruff check src/sol_execbench/core/scoring/amd_bound_graph.py src/sol_execbench/core/scoring/amd_bound_estimates.py src/sol_execbench/core/scoring/amd_sol.py src/sol_execbench/core/scoring/__init__.py tests/sol_execbench/test_amd_bound_estimates.py tests/sol_execbench/test_amd_bound_graph.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py
```

Result recorded in `43-VERIFICATION.md`: all checks passed.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-23 | 10 | 10 | 0 | Codex / GSD secure-phase |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-23
