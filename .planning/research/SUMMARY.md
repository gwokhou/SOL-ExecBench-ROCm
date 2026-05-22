# Project Research Summary

**Project:** SOL ExecBench ROCm Port
**Domain:** SOL ExecBench ROCm engineering-practice adaptation
**Researched:** 2026-05-22
**Confidence:** HIGH

## Executive Summary

`hip-execbench` is strongest as an engineering workflow around HIP kernel
development: explicit pipeline stages, compile-cache discipline, richer
diagnostics, profiling/report output, agent-readable summaries, and repeated-run
baseline comparison. Those are valuable practices, but the implementation is
not semantically equivalent to the current SOL ExecBench ROCm harness. The
current repository has stronger paper-aligned correctness, timing, trace, and
reward-hack behavior.

The recommended v1.4 approach is to adapt practices, not frameworks. Keep the
existing CLI, Pydantic schemas, solution format, trace JSONL, and eval driver as
the public benchmark contract. Add internal or opt-in helpers for stage
diagnostics, validation readiness, derived evidence reports, and compatibility
guardrails.

The main risk is claim inflation: CDNA 3 validation readiness can be useful
without proving real CDNA 3 hardware validation. v1.4 should implement CDNA 3
readiness and evidence collection, but only RDNA 4 is required to receive real
unit and E2E validation evidence in this milestone.

## Key Findings

### Recommended Stack

No new runtime dependency is recommended. Use the existing Python package,
Pydantic v2 schemas, Click/Rich CLI surface, PyTorch ROCm event timing, JSON and
Markdown derived artifacts, and pytest validation. Do not import the
TypeScript/Node stack from `hip-execbench`.

**Core technologies:**
- Python/Pydantic: preserve public benchmark contracts.
- PyTorch ROCm event timing: preserve current timing semantics.
- Pytest: enforce public contract, readiness, and RDNA 4 E2E validation.
- JSON/Markdown: produce separate evidence artifacts without mutating traces.

### Expected Features

**Must have:**
- Compatibility contract inventory and guardrail tests.
- Source-grounded `hip-execbench` practice adaptation map.
- Internal/additive stage diagnostics or evidence model.
- CDNA 3 validation readiness workflow for `gfx94*`.
- RDNA 4 unit and E2E validation evidence.

**Should have:**
- Evidence/report writer derived from existing trace and diagnostics.
- Agent-readable summaries that do not replace trace JSONL.
- Profiling readiness classification.

**Defer:**
- Public experimental CLI.
- AMD-native SOL interpretation model.
- Real CDNA 3 hardware validation pass and claim update.

### Architecture Approach

Add a new internal/additive layer around the existing eval path. The layer can
observe and summarize parse/package/compile/eval/verify/timing stages, classify
validation readiness, and generate evidence files. It must not replace
`ProblemPackager`, `build_ext.py`, `eval_driver.py`, or trace JSONL as the
benchmark authority.

**Major components:**
1. Compatibility inventory and tests — define non-negotiable public contracts.
2. Stage diagnostics/evidence helpers — capture status and failure context.
3. Validation readiness helpers — model RDNA 4/CDNA 3 requirements and evidence.
4. Derived reporting — generate optional JSON/Markdown summaries.
5. RDNA 4 validation closure — run unit and E2E checks.

### Critical Pitfalls

1. **Breaking paper harness semantics** — avoid by keeping new helpers
   observational or additive.
2. **Treating readiness as CDNA 3 validation** — avoid through explicit claim
   levels and docs/tests.
3. **Importing dependencies instead of practices** — avoid TypeScript/Node
   runtime additions.
4. **Report format becoming accidental API** — keep reports derived and
   separate from trace JSONL.
5. **Narrow RDNA 4 validation** — require unit plus E2E evidence.

## Implications for Roadmap

### Phase 1: Compatibility and Practice Inventory

**Rationale:** Establish what cannot change before adapting patterns.
**Delivers:** Source-grounded adaptation map and public-contract guardrails.
**Addresses:** Compatibility and practice selection.
**Avoids:** Public contract drift.

### Phase 2: Internal Diagnostics and Evidence Model

**Rationale:** Stage-result and evidence helpers are the safest high-value
`hip-execbench` adaptation.
**Delivers:** Internal stage diagnostics and derived evidence structures.
**Uses:** Existing Python modules and pytest.
**Implements:** Additive layer around existing benchmark flow.

### Phase 3: CDNA 3 Validation Readiness

**Rationale:** User wants implementation without real CDNA 3 hardware validation.
**Delivers:** `gfx94*` readiness checks, command/evidence template, claim
guardrails, and tests.
**Avoids:** Claim inflation.

### Phase 4: RDNA 4 Validation Closure

**Rationale:** User requires actual RDNA 4 unit and E2E validation.
**Delivers:** Unit test pass, E2E evidence, and final compatibility checks.
**Avoids:** Looks-done-but-isn't validation.

### Phase Ordering Rationale

- Compatibility comes before implementation so borrowed practices cannot drift
  public contracts.
- Diagnostics/evidence comes before CDNA readiness because readiness needs a
  place to record evidence.
- RDNA 4 closure comes last because it validates the integrated result.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** Exact internal API shape for stage diagnostics.
- **Phase 3:** CDNA 3 evidence template and claim-level wording.
- **Phase 4:** E2E command selection based on available RDNA 4 environment.

Phases with standard patterns:
- **Phase 1:** Existing public-contract guardrail pattern is already present.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Existing Python stack is clearly sufficient. |
| Features | HIGH | User constraints and source comparison point to the same priorities. |
| Architecture | HIGH | Existing public contracts require an additive/internal layer. |
| Pitfalls | HIGH | Risks are directly visible in current and comparison code paths. |

**Overall confidence:** HIGH

### Gaps to Address

- Exact validation evidence file format should be decided in phase planning.
- Whether to add a new script or keep validation readiness as internal helpers
  should be decided based on compatibility and testability.
- E2E evidence depends on the local RDNA 4 environment available at execution
  time.

## Sources

### Primary

- `src/sol_execbench/core/data/*.py` — public schema contracts.
- `src/sol_execbench/driver/templates/eval_driver.py` — paper-style correctness and reward-hack harness.
- `src/sol_execbench/core/bench/timing.py` — timing semantics.
- `~/PyCharmMiscProject/hip-playground/hip-execbench/src/pipeline/runner.ts` — stage-result pattern.
- `~/PyCharmMiscProject/hip-playground/hip-execbench/src/compiler/hipcc.ts` — compile-cache discipline.
- `~/PyCharmMiscProject/hip-playground/hip-execbench/src/baseline/comparator.ts` — repeated-run comparison pattern.

---
*Research completed: 2026-05-22*
*Ready for roadmap: yes*
