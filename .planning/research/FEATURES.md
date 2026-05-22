# Feature Research

**Domain:** SOL ExecBench ROCm engineering-practice adaptation
**Researched:** 2026-05-22
**Confidence:** HIGH

## Feature Landscape

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Compatibility-preserving adaptation map | User explicitly wants `hip-execbench` lessons without public interface breakage. | MEDIUM | Must distinguish accepted, rejected, and deferred practices with code evidence. |
| Internal stage diagnostics | `hip-execbench` stage results make partial failures easier to understand. | MEDIUM | Additive helpers should wrap existing subprocess/eval outcomes without changing trace JSONL. |
| CDNA 3 validation readiness | Current project defers real CDNA 3 validation but needs executable readiness. | MEDIUM | Implement detection, command/evidence plan, failure classification, and no-claim guardrails. |
| RDNA 4 unit + E2E validation | User requires actual validation on RDNA 4 for v1.4 changes. | HIGH | Must include focused unit tests plus E2E command evidence. |
| Public-contract guardrails | Existing schemas and CLI are compatibility commitments. | MEDIUM | Tests should fail if new helpers mutate public schema, CLI behavior, or trace fields. |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Evidence bundle/report writer | Makes validation outcomes reusable for PRs and future CDNA 3 runs. | MEDIUM | Keep opt-in/internal and write separate artifact files. |
| Baseline/statistical comparison readiness | Learns from `hip-execbench` significance discipline. | HIGH | Should remain advisory unless repeated-sample contract exists. |
| Compile/eval stage manifest | Improves debugging without replacing the eval driver. | MEDIUM | Can capture commands, status, duration, paths, and logs internally. |
| Agent-readable summary over existing traces | Helps automation consume results. | MEDIUM | Must be derived from trace JSONL, not a replacement trace contract. |
| Profiling readiness classification | Guides operator on `rocprofv3`/architecture support. | MEDIUM | Existing diagnostics can be expanded without adding mandatory profiling. |

### Anti-Features

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Direct TypeScript framework import | `hip-execbench` has useful modules. | Creates a second runtime and incompatible schemas. | Reimplement selected patterns in Python internal helpers. |
| Replacing trace JSONL with agent JSON | Agent output is convenient. | Breaks benchmark machine-readable contract. | Derive optional agent/evidence summaries from existing traces. |
| Treating readiness as hardware validation | It closes a visible deferred gap superficially. | Produces false CDNA 3 claims. | Separate readiness status from hardware-pass evidence. |
| Replacing current timing with binary-reported timing | Standalone HIP timing is convenient. | Loses current L2 clearing, unique data pointer, and reference semantics. | Keep current timing; add optional diagnostics around it. |

## Feature Dependencies

```
Compatibility Contract Inventory
    requires -> Public Contract Guardrail Tests
    requires -> Adaptation Map

Stage Diagnostics
    requires -> Compatibility Contract Inventory
    enhances -> Evidence Bundle

CDNA 3 Validation Readiness
    requires -> Environment Detection
    requires -> Evidence Bundle
    requires -> No-Claim Guardrails

RDNA 4 Unit + E2E Validation
    validates -> Stage Diagnostics
    validates -> CDNA 3 Readiness fallback behavior
    validates -> Compatibility Guardrails
```

## MVP Definition

### Launch With v1.4

- [ ] Compatibility contract inventory for CLI, schemas, trace JSONL, solution format, and eval semantics.
- [ ] `hip-execbench` engineering practice analysis grounded in source code.
- [ ] Internal/additive stage diagnostics or evidence model.
- [ ] CDNA 3 validation readiness workflow that can run detection and produce evidence templates without real `gfx94*`.
- [ ] RDNA 4 unit and E2E validation evidence for the implemented path.

### Add After Validation

- [ ] Opt-in agent-readable report derived from existing trace JSONL.
- [ ] Repeated-sample baseline comparison with explicit statistical assumptions.
- [ ] Compile cache only if it can be proven not to weaken correctness or build reproducibility.

### Future Consideration

- [ ] Public experimental CLI for validation evidence generation.
- [ ] AMD-native SOL interpretation model.
- [ ] Real CDNA 3 full adapted suite pass and claim update.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Compatibility contract guardrails | HIGH | MEDIUM | P1 |
| Engineering practice adaptation map | HIGH | LOW | P1 |
| CDNA 3 readiness workflow | HIGH | MEDIUM | P1 |
| RDNA 4 unit + E2E validation | HIGH | HIGH | P1 |
| Internal stage diagnostics | MEDIUM | MEDIUM | P2 |
| Evidence/report writer | MEDIUM | MEDIUM | P2 |
| Statistical baseline extension | MEDIUM | HIGH | P3 |

## Sources

- Current repository `src/sol_execbench/driver/templates/eval_driver.py` for paper-style harness behavior.
- Current repository `tests/sol_execbench/test_public_contract_guardrails.py` for prior compatibility test pattern.
- `hip-execbench/src/pipeline/runner.ts` for stage-result and partial-result structure.
- `hip-execbench/src/baseline/comparator.ts` for repeated-run/statistical comparison discipline.
- `hip-execbench/src/cli/commands/score.ts` for agent/report/profile output shape.

---
*Feature research for: v1.4 engineering-practice adaptation*
*Researched: 2026-05-22*
