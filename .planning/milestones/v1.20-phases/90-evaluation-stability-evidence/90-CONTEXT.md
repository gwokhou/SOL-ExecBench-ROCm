# Phase 90 Context: Evaluation Stability Evidence

## Objective

Add a diagnostic-only `evaluation_stability.v1` sidecar that summarizes timing
quality from existing timing evidence. The feature must not change canonical
trace JSONL, correctness, timing, score, or evaluator semantics.

## Existing Code Context

- `src/sol_execbench/core/bench/rocm_profiler.py` already defines
  `Rocprofv3TimingEvidence.to_dict()` with backend, activity domain, warmup,
  iterations, trial count, clock lock state, fallback flags, kernel duration,
  and parsed rows.
- `src/sol_execbench/core/bench/timing_policy.py` defines backend and activity
  domain semantics.
- v1.19 reporting modules use strict Pydantic sidecars, deterministic checksum
  generation, JSON/Markdown writers, standalone scripts, and public contract
  guardrails.
- `scripts/run_dataset.py` already stores timing evidence refs in execution
  closure records when `--timing-evidence-dir` is supplied.

## Constraints

- Sidecar-only. Do not add primary `sol-execbench` CLI options or alter public
  Trace, Workload, Definition, Solution, score, timing, or evaluator contracts.
- CPU-safe core tests should cover classification and deterministic
  serialization.
- Use existing timing evidence payloads and source refs. Do not require live
  profiler collection for the basic report.
- Claim boundary remains explicit: stability supports interpretation only and
  does not create correctness, score, paper-parity, native-host, or leaderboard
  authority.

## Planned Shape

- Core module: `src/sol_execbench/core/evaluation_stability.py`
- Script: `scripts/report_evaluation_stability.py`
- Tests:
  - `tests/sol_execbench/test_evaluation_stability.py`
  - `tests/sol_execbench/test_evaluation_stability_script.py`
  - public contract guardrail additions

