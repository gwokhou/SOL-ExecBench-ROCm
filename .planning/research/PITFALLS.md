# Pitfalls Research

**Domain:** SOL ExecBench ROCm engineering-practice adaptation
**Researched:** 2026-05-22
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Breaking Paper Harness Semantics

**What goes wrong:**
New pipeline helpers replace or bypass reference execution, 10-round correctness,
timing integrity, L2 clearing, unique tensor allocation, or reward-hack checks.

**Why it happens:**
`hip-execbench` has a cleaner standalone HIP binary flow, but that flow is not
semantically equivalent to the current SOL ExecBench eval driver.

**How to avoid:**
Keep new helpers observational or additive. Any execution-path change must prove
equivalence through focused tests and E2E validation.

**Warning signs:**
New code emits benchmark results without going through `eval_driver.py`, or trace
schema fields start depending on new report helpers.

**Phase to address:**
First compatibility/contract phase.

---

### Pitfall 2: Treating CDNA 3 Readiness as CDNA 3 Validation

**What goes wrong:**
Docs or reports imply `gfx94*` passed even though only readiness or command
construction was tested.

**Why it happens:**
Validation readiness is useful and can look like completion in status summaries.

**How to avoid:**
Introduce explicit claim levels and require real hardware evidence before
changing CDNA 3 validation status.

**Warning signs:**
Phrases like "CDNA 3 validated" appear without a recorded full-suite `gfx94*`
run.

**Phase to address:**
Validation readiness phase.

---

### Pitfall 3: Importing `hip-execbench` Dependencies Instead of Practices

**What goes wrong:**
The Python package gains a Node/TypeScript or frontend reporting dependency
surface.

**Why it happens:**
The useful patterns in `hip-execbench` are packaged in a different stack.

**How to avoid:**
Translate patterns into existing Python modules using dataclasses, JSON,
Markdown, pytest, and current CLI conventions.

**Warning signs:**
`package.json`, TypeScript build steps, or report frontend assets become runtime
requirements for `sol-execbench`.

**Phase to address:**
Practice adaptation phase.

---

### Pitfall 4: Report Format Becomes Accidental Public API

**What goes wrong:**
Derived evidence files become relied upon as canonical benchmark output before
their stability is defined.

**Why it happens:**
Agent-readable JSON and Markdown reports are useful in automation.

**How to avoid:**
Label derived artifacts as evidence/reporting outputs and keep trace JSONL as
the benchmark output contract.

**Warning signs:**
Tests assert new report fields as if they are trace schema fields.

**Phase to address:**
Reporting/evidence phase.

---

### Pitfall 5: RDNA 4 Validation is Too Narrow

**What goes wrong:**
Only unit tests pass, but the integrated benchmark flow regresses.

**Why it happens:**
Additive helpers can still affect environment detection, packaging, or scripts.

**How to avoid:**
Require both focused unit tests and RDNA 4 E2E evidence for implemented paths.

**Warning signs:**
No E2E run is recorded after changing validation or diagnostics code.

**Phase to address:**
Final validation closure phase.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Add new trace fields for reports | Easy data plumbing | Breaks public contract | Never in v1.4 |
| Mock all hardware checks | Fast tests | Readiness not grounded in actual ROCm behavior | Acceptable only for unit tests with separate RDNA 4 E2E |
| Copy TypeScript behavior verbatim | Faster initial implementation | Mismatched semantics and dependencies | Never for public benchmark path |
| Skip claim-level docs | Less writing | Future false validation claims | Never |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| ROCm detection | Treat missing tools as generic failure | Return stage-aware readiness diagnostics and hints. |
| Evidence writing | Write into trace JSONL | Write separate evidence files. |
| Baseline comparison | Apply significance tests to single samples | Require repeated timing samples before statistical claims. |
| Compile caching | Cache without all behavior-affecting inputs | Include source, flags, ROCm version, target arch, and invalidate conservatively. |

## "Looks Done But Isn't" Checklist

- [ ] **CDNA 3 readiness:** Has commands and evidence template, but docs still say real validation is deferred.
- [ ] **RDNA 4 validation:** Unit tests pass and E2E evidence is recorded.
- [ ] **Compatibility:** CLI help, schema behavior, and trace JSONL remain unchanged.
- [ ] **Practice adaptation:** Each borrowed pattern has accepted/rejected/deferred rationale.
- [ ] **Scoring/baseline:** Any AMD performance claim includes guardrail language.

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Breaking harness semantics | Compatibility inventory and guardrails | Public contract tests and E2E trace comparison. |
| Readiness claim inflation | CDNA 3 readiness implementation | Tests assert no CDNA 3 validation claim without evidence. |
| Dependency creep | Practice adaptation | Dependency diff and pyproject audit. |
| Accidental public report API | Evidence/reporting implementation | Tests prove trace schema is unchanged. |
| Narrow RDNA 4 validation | Validation closure | Recorded unit + E2E command outcomes. |

## Sources

- Current eval driver and timing helpers.
- Current public contract guardrail tests.
- `hip-execbench` pipeline, scoring, baseline, reporting, and profiler modules.
- User constraints from v1.4 milestone questioning.

---
*Pitfalls research for: v1.4 engineering-practice adaptation*
*Researched: 2026-05-22*
