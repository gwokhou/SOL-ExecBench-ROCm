# Phase 19: Compatibility and Practice Inventory - Plan

**Status:** Ready for execution  
**Created:** 2026-05-22  
**Requirements:** COMPAT-01, COMPAT-02, COMPAT-03

## Objective

Create a source-grounded compatibility inventory and guardrail suite for v1.4 so
later phases can adapt hip-execbench engineering practices without changing SOL
ExecBench ROCm public interfaces or benchmark semantics.

## Scope

In scope:

- Document the current public contracts with source references.
- Expand the hip-execbench practice map into accepted, rejected, and deferred
  categories with implementation evidence.
- Add or strengthen tests that protect the inventory and public-contract
  boundary.

Out of scope:

- No new public CLI flags or commands.
- No Pydantic schema field changes.
- No trace JSONL field changes.
- No benchmark execution behavior changes.
- No CDNA3 or RDNA4 hardware validation in this phase.

## Threat Model

| Threat | Severity | Mitigation |
|--------|----------|------------|
| T-19-01: Inventory accidentally blesses a public schema/CLI change. | High | Tests assert current CLI and schema contract examples remain valid. |
| T-19-02: hip-execbench practice is documented as accepted without source evidence. | Medium | Practice map tests require source-path evidence and accepted/rejected/deferred categories. |
| T-19-03: Derived reporting ideas are confused with canonical trace JSONL. | High | Inventory and tests state trace JSONL remains authoritative and derived reports must stay separate. |

## Wave 1: Inventory And Guardrails

### Task 19-01: Add Public Contract Inventory

**Requirement:** COMPAT-01  
**Files:**

- `docs/internal/v1_4_compatibility_inventory.md`

**Steps:**

1. Create an internal inventory document.
2. Record source-backed public contracts for:
   - `sol-execbench` CLI behavior.
   - `Solution`, `Workload`, `Definition`, and `Trace` Pydantic models.
   - Solution source format and supported ROCm metadata.
   - Trace JSONL stdout/output behavior.
   - Eval-driver status and benchmark semantics.
3. Include non-goals that explicitly forbid public API/schema/trace changes in
   this phase.

**Verification:**

```bash
uv run pytest tests/sol_execbench/test_public_contract_guardrails.py
```

### Task 19-02: Refine hip-execbench Practice Classification

**Requirement:** COMPAT-03  
**Files:**

- `docs/internal/hip_execbench_practice_map.md`

**Steps:**

1. Split the practice map into explicit accepted, rejected, and deferred
   sections.
2. For every accepted practice, cite the hip-execbench source path and explain
   the SOL ExecBench ROCm adaptation boundary.
3. For rejected/deferred practices, tie the decision to compatibility,
   benchmark semantics, dependency surface, or insufficient repeated-sample
   contract.
4. Preserve the current baseline comparison guidance.

**Verification:**

```bash
uv run pytest tests/sol_execbench/test_hip_execbench_practice_map.py
```

### Task 19-03: Strengthen Guardrail Tests

**Requirement:** COMPAT-02  
**Files:**

- `tests/sol_execbench/test_public_contract_guardrails.py`
- `tests/sol_execbench/test_hip_execbench_practice_map.py`

**Steps:**

1. Add tests that require the compatibility inventory to mention all public
   contract categories and source files.
2. Add tests that fail if the practice map no longer contains accepted,
   rejected, and deferred classifications with source evidence.
3. Add tests that keep normal `sol-execbench` help free of Phase 19 diagnostic
   or hip-execbench imports.
4. Confirm no production runtime files changed in this phase.

**Verification:**

```bash
uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_hip_execbench_practice_map.py
uv run ruff check tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_hip_execbench_practice_map.py
```

## Completion Criteria

- COMPAT-01: Inventory exists and is source-grounded.
- COMPAT-02: Guardrail tests fail on public contract drift.
- COMPAT-03: hip-execbench practices are classified as accepted, rejected, or
  deferred with rationale.
- No source runtime module, public CLI, schema, trace, or eval-driver contract is
  changed.
