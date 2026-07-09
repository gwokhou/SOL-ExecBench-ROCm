# Coupling Governance

This project treats coupling refactors as bounded maintenance work, not an
open-ended chase for lower import counts.

## Convergence Criteria

The source tree is considered converged for P0/P1 coupling when all of these
checks pass:

```bash
uv run pytest tests/sol_execbench/cli/test_module_boundaries.py -q
uv run python scripts/check_coupling.py
uv run --with ruff ruff check .
```

The coupling check requires:

- no internal import strongly connected components;
- no internal source imports from broad compatibility facades;
- P0/P1 orchestration modules within their line-count and fanout bounds;
- generated evaluation driver imports routed through
  `sol_execbench.driver.eval_runtime_api`.

## Allowed High Inbound Modules

High inbound imports are acceptable for stable schema and domain model modules.
These modules should stay low-level and must not depend on CLI, dataset runner,
driver orchestration, or benchmark runtime layers.

Examples:

- `sol_execbench.core.data.definition`
- `sol_execbench.core.data.workload`
- `sol_execbench.core.data.trace`
- `sol_execbench.core.scoring.amd_bound_graph_models`
- `sol_execbench.core.scoring.amd_hardware_models`

Do not split these modules only because their inbound count is high. Refactor
them only when they gain higher-layer dependencies, unrelated responsibilities,
or repeated change conflicts.

## Facade Policy

Compatibility facades are allowed for external users and old import paths.
Internal source code should import the focused owning module instead.

Rules:

- `sol_execbench.core.dataset` remains a lazy compatibility facade.
- New internal code imports from submodules such as
  `sol_execbench.core.dataset.sharding`.
- `sol_execbench.core` and `sol_execbench.core.data` are not internal import
  roots for implementation code.
- `sol_execbench.core.scoring` package re-exports are not used by internal
  implementation modules.

## Orchestration Policy

Orchestration modules may coordinate multiple subsystems, but they should not
own subsystem details.

Current P0/P1 boundaries:

- `sol_execbench.driver.templates.eval_driver` imports only
  `sol_execbench.driver.eval_runtime_api`.
- `sol_execbench.core.scoring.amd_score_reports` assembles reports; derived
  sidecar and artifact resolution lives in
  `sol_execbench.core.scoring.amd_score_derived_artifacts`.
- `sol_execbench.driver.problem_packager` coordinates packaging; build flag
  logic lives in `sol_execbench.driver.build_config`, and file staging lives in
  `sol_execbench.driver.staging`.
- `sol_execbench.core.bench.rocm_profiler` stays a public entrypoint; command,
  artifact, profile, and timing details live in focused profiler modules.

## When To Continue Refactoring

Continue coupling refactors only when one of these happens:

- `scripts/check_coupling.py` fails;
- `tests/sol_execbench/cli/test_module_boundaries.py` fails;
- a P0/P1 module exceeds its fanout or line-count boundary;
- implementation code imports from a broad compatibility facade;
- a stable model starts depending on a higher layer;
- a new feature repeatedly requires unrelated changes across multiple modules.

Otherwise, stop. Passing guardrails mean coupling is converged for the current
scope.
