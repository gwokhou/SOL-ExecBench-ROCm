# Architecture navigation for maintainers and coding agents

This map identifies ownership and local verification boundaries. Artifact JSON is
untrusted at parser and compatibility adapters; domain services consume typed
models or request views after that boundary.

| Domain | Owner module | Main entry | Data flow | Key invariant | Local checks |
| --- | --- | --- | --- | --- | --- |
| CLI evaluation | `cli/evaluation` | `EvaluationRequest` → `run_evaluation_cli` | resolve/load → stage/compile → runtime → sidecars/result | `ProblemPackager.close()` runs on every exit path; CLI exit and JSON formats remain stable | `tests/sol_execbench/cli`, module-boundary tests |
| Generated driver | `core/bench` | `WorkloadEvaluationRequest` → `evaluate_workloads` | integrity gate → correctness → timing → trace | injected functions live in `EvaluationDependencies`; workload data is separate | `tests/sol_execbench/core/bench`, driver tests |
| AMD bound sanity | `core/scoring/amd_bound_sanity` | `SanityInputs` → `build_amd_bound_sanity_report` | ingest → audit → aggregate → report/checksum | raw mappings stop at input/ingest adapters; workload state is typed | AMD sanity and v1.19 evidence tests |
| Fusion validation | `core/scoring/fusion_validation` | `fusion_validation_from_dict`, collection builders | parse → resource evidence → policy → artifact | schema parsing is strict and group ordering deterministic | fusion validation tests |
| Official score | `core/scoring/official_score` | official/suite evidence builders | validate refs/checksums → authority blockers → aggregate | scoring formula, blocker literals and checksum semantics are stable | official score evidence/CLI tests |
| Release baseline | `core/scoring/release_baseline` | build/verify/publication services | collect → verify authority → publish/compare | only immutable, checksum-bound evidence may gain authority | release baseline tests |

Run `scripts/check_readability.py`, `scripts/check_coupling.py`, Ruff, and `ty`
after moving an entry point or artifact boundary. Hardware-marked tests remain
evidence checks and must not be replaced by mocks.
