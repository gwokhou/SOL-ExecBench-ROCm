# Phase 19 Research: Compatibility and Practice Inventory

## RESEARCH COMPLETE

**Phase:** 19 - Compatibility and Practice Inventory  
**Date:** 2026-05-22  
**Mode:** Inline autonomous research from local source code

## Goal

Plan Phase 19 from source evidence, not README claims. The phase should produce
an explicit compatibility boundary, classify hip-execbench engineering practices,
and add guardrails that prevent accidental public contract drift.

## Source Evidence

### SOL ExecBench ROCm Public Contracts

- CLI contract is in `src/sol_execbench/cli/main.py`.
  - Current public options include `--definition`, `--workload`, `--solution`,
    `--config`, `--out`, `--json`, `--lock-clocks`, `--timeout`, and
    `--verbose`.
  - Positional `PROBLEM_DIR` mode remains supported.
  - Normal benchmark runs parse eval-driver stdout as trace JSONL.
- Solution schema is in `src/sol_execbench/core/data/solution.py`.
  - ROCm-facing language/hardware support is represented through Pydantic
    models and strings such as `hip_cpp`, `gfx1200`, and `gfx942`.
  - Source-file path validation and reward-hack checks are part of the contract
    boundary.
- Workload schema is in `src/sol_execbench/core/data/workload.py`.
  - `uuid`, `axes`, and `inputs` are existing compatibility points.
- Trace schema is in `src/sol_execbench/core/data/trace.py`.
  - Workload-only traces use `solution=None` and `evaluation=None`.
  - Evaluation traces use `EvaluationStatus`, `Correctness`, `Performance`,
    and `Environment`.
- Eval-driver behavior is in
  `src/sol_execbench/driver/templates/eval_driver.py`.
  - It writes JSONL trace objects to stdout.
  - It emits existing status values for reward hack, runtime, correctness,
    timing, and pass/fail paths.
  - Comments explicitly reserve trace output behavior; debug/log output must not
    corrupt JSONL stdout.

### Existing Guardrails

- `tests/sol_execbench/test_public_contract_guardrails.py` already protects
  schema shapes, CLI help options, HIP-facing examples, and deferred CDNA3
  hardware validation language.
- `tests/sol_execbench/test_hip_execbench_practice_map.py` already checks the
  practice map for baseline comparison, contract-changing rejected imports, and
  guardrail language.

### hip-execbench Practices Worth Adapting

- `src/profiler/router.ts` has a pure backend-selection function that returns
  backend, reason, fallback status, and effective level. This is a strong model
  for internal diagnostics because it makes readiness and fallback decisions
  explicit without changing benchmark semantics.
- `src/errors/index.ts` uses a typed error hierarchy with fields such as path,
  field, expected, actual, component, detail, and fix hints. This is useful as
  an internal model for more actionable stage diagnostics.
- `src/agent/builder.ts` transforms pipeline internals into stable
  machine-readable summaries through pure helper functions. SOL ExecBench ROCm
  should adapt the transformation-layer idea without adding fields to trace
  JSONL.
- `src/baseline/comparator.ts` performs named-source baseline comparison with
  win/parity/loss-style thresholds, repeated runs, hardware context, and
  pairwise statistical tests. SOL ExecBench ROCm already has a trace-file
  baseline CLI; Phase 19 should document the compatibility boundary around it.
- `src/pipeline/statistics.ts` implements Mann-Whitney U without extra runtime
  dependencies. This is a future candidate only when SOL ExecBench ROCm has a
  repeated-sample trace contract.

### hip-execbench Practices To Reject Or Defer

- `src/schemas/*.ts` uses Zod and TypeScript schemas. Importing this model would
  change SOL ExecBench ROCm's public Pydantic validation surface.
- `src/cli/index.ts` and `src/main.ts` expose a distinct CLI shape. Replacing
  `sol-execbench` with `hip-bench`-style subcommands would break users.
- `src/reporting/html.ts` and `src/reporting/charts.ts` imply HTML/Plotly
  reporting dependencies and a new report artifact surface. That is outside
  Phase 19.
- `src/agent/builder.ts`'s agent document is useful as a design pattern but must
  not become normal-run trace JSONL output without a public schema decision.
- Directly importing repeated-run significance tests is premature until the
  benchmark execution contract records enough repeated samples to justify the
  statistic.

## Planning Guidance

- Deliver an internal compatibility inventory doc with concrete source refs for
  CLI, schemas, trace JSONL, solution format, and eval-driver behavior.
- Update the practice map to classify accepted, rejected, and deferred practices
  explicitly, with source paths and compatibility rationale.
- Extend guardrail tests so they fail if the inventory/practice map stops
  protecting the public contracts or if Phase 19 adds production runtime/API
  changes.
- Keep this phase to documentation and tests unless a small test helper is
  clearly necessary.

## Validation Architecture

- Unit tests:
  - Assert compatibility inventory sections exist for CLI, solution schema,
    workload schema, trace JSONL, eval-driver behavior, and non-goals.
  - Assert practice classifications include accepted, rejected, and deferred
    categories tied to source paths.
  - Assert the public CLI still does not expose diagnostics/profile/imported
    hip-execbench commands.
- Regression checks:
  - `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py`
  - `uv run pytest tests/sol_execbench/test_hip_execbench_practice_map.py`
  - `uv run ruff check docs/internal tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_hip_execbench_practice_map.py`

## Risks

- Over-documenting future phases as already implemented. Mitigation: separate
  accepted practices from current implementation state.
- Treating hip-execbench's agent or report contracts as compatible public output.
  Mitigation: keep trace JSONL explicitly authoritative.
- Making CDNA3 validation claims before hardware execution. Mitigation: only
  state implementation/readiness for later phases and preserve deferred
  validation language.
