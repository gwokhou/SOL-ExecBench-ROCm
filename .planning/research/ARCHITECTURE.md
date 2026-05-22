# Architecture Research

**Domain:** SOL ExecBench ROCm engineering-practice adaptation
**Researched:** 2026-05-22
**Confidence:** HIGH

## Recommended Architecture

### System Overview

```
Existing public surface
    sol-execbench CLI
    Pydantic definition/workload/solution/trace schemas
    Trace JSONL output
        |
        v
Existing eval core
    ProblemPackager -> build_ext.py -> eval_driver.py
    reference correctness, timing, reward-hack checks
        |
        v
New internal/additive layer
    stage diagnostics
    validation readiness
    evidence/report helpers
    compatibility guardrails
```

The new layer should observe, summarize, and validate existing behavior. It
must not become the primary benchmark contract.

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| Compatibility inventory | Define what must not change. | Tests and docs referencing CLI help, schemas, trace fields, and solution model. |
| Stage diagnostics | Capture parse/package/compile/eval/verify/timing readiness and failure context. | Internal dataclasses and pure helpers under `src/sol_execbench/core/`. |
| Validation readiness | Model RDNA 4 and CDNA 3 validation preconditions and evidence requirements. | Internal Python helpers plus scripts/docs/tests. |
| Evidence writer | Produce opt-in JSON/Markdown evidence files. | Derived artifacts separate from trace JSONL. |
| RDNA 4 validation tests | Prove implementation works on validated platform. | Unit tests plus E2E command/test evidence. |

## Recommended Project Structure

```
src/sol_execbench/core/
├── diagnostics.py              # Existing diagnostics helpers, extend carefully
├── reporting.py                # Existing pure trace summaries, extend only as derived reports
├── validation.py               # Candidate new internal validation readiness helpers
└── baseline.py                 # Existing trace-baseline comparison, keep contract stable

scripts/
└── validate_rocm.py            # Candidate opt-in validation/evidence runner if needed

tests/sol_execbench/
├── test_public_contract_guardrails.py
├── test_rocm_diagnostics_reporting.py
├── test_validation_readiness.py
└── test_e2e.py

docs/
└── validation.md               # Candidate evidence and claim-boundary documentation
```

### Structure Rationale

- Keep public CLI and schemas in place.
- Put new functionality in internal modules first.
- Use docs/tests to expose behavior before adding any public command.
- Keep validation evidence separate from trace JSONL.

## Architectural Patterns

### Pattern 1: Stage Result Wrapper

**What:** Represent each phase result as status, duration, error, and optional data.
**When to use:** Internal command readiness, validation workflows, and evidence generation.
**Trade-offs:** Improves diagnostics but can become a shadow pipeline if allowed to replace eval semantics.

### Pattern 2: Derived Report Layer

**What:** Generate JSON/Markdown summaries from existing trace objects, state, and diagnostics.
**When to use:** Agent/operator readability.
**Trade-offs:** Safe if derived; unsafe if users start treating it as canonical trace schema.

### Pattern 3: Claim-Level Guardrails

**What:** Explicitly classify readiness, simulated/unsupported checks, RDNA 4 evidence, and real CDNA 3 validation.
**When to use:** Any score, baseline, or validation evidence involving AMD architecture claims.
**Trade-offs:** Adds text/tests, but prevents false hardware validation claims.

## Data Flow

### Validation Readiness Flow

```
Environment detection
    -> classify gfx target
    -> select validation profile
    -> run or skip hardware-bound checks
    -> collect evidence metadata
    -> write separate evidence artifact
    -> summarize claim level
```

### Engineering Adaptation Flow

```
hip-execbench source pattern
    -> classify as accepted/rejected/deferred
    -> map to internal Python adaptation
    -> add guardrail tests
    -> validate RDNA 4 path
```

## Integration Points

| Boundary | Communication | Notes |
|----------|---------------|-------|
| CLI -> ProblemPackager | Existing direct calls | Avoid changing arguments or exit behavior. |
| ProblemPackager -> eval driver | Existing staging files | New diagnostics can observe, not replace. |
| Trace JSONL -> reports | Derived object loading | Reports must not add fields to trace JSONL. |
| Validation helpers -> docs/tests | Python helper APIs | Keep hardware claims explicit. |

## Anti-Patterns

### Shadow Benchmark Pipeline

**What people do:** Add a new pipeline that compiles/runs kernels differently from `sol-execbench`.
**Why it is wrong:** Creates incompatible results and weakens paper semantics.
**Do this instead:** Wrap existing stages with diagnostics and evidence.

### Public Contract Drift

**What people do:** Add convenient fields to trace or solution models for new reports.
**Why it is wrong:** Breaks dataset and tool compatibility.
**Do this instead:** Generate separate derived artifacts.

### Readiness Claim Inflation

**What people do:** Treat CDNA 3 readiness tests as validation.
**Why it is wrong:** No real `gfx94*` hardware evidence exists.
**Do this instead:** Use claim levels: not-run, readiness-only, RDNA-validated, CDNA-validated.

## Sources

- `src/sol_execbench/cli/main.py` and `src/sol_execbench/driver/problem_packager.py` for current pipeline boundaries.
- `src/sol_execbench/driver/templates/eval_driver.py` for benchmark semantics.
- `hip-execbench/src/pipeline/runner.ts` for stage result architecture.
- `hip-execbench/src/cli/commands/score.ts` for derived agent/report architecture.

---
*Architecture research for: v1.4 engineering-practice adaptation*
*Researched: 2026-05-22*
