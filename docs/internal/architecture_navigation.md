# Architecture navigation for maintainers

Raw JSON/YAML is validated at parser boundaries. Domain stages consume typed
models or typed request/result objects.

| Domain | Owning entry | Flow | Invariant | Focused checks |
| --- | --- | --- | --- | --- |
| CLI evaluation | `src/sol_execbench/cli/evaluation/evaluator.py` | resolve → package → run → parse → derive/report | staging closes on every path; relative metrics are outer-process work | `tests/sol_execbench/cli/evaluation/` |
| Process staging | `src/sol_execbench/driver/problem_packager.py` | normalize → compile assets → stage three runtime templates | execute target is the orchestrator | `tests/sol_execbench/driver/test_problem_packager.py` |
| Trusted reference | `src/sol_execbench/core/bench/reference_service.py` | load reference/input sources → produce case/timing → safetensors response | no candidate import; failures are structured | reference protocol and driver tests |
| Candidate execution | `src/sol_execbench/core/bench/eval_workload_execution.py` | transferred case → correctness → integrity → timing → Trace | no reference load/call; timed outputs validated | `tests/sol_execbench/driver/test_eval_driver.py` |
| Evaluation diagnostics | `src/sol_execbench/core/bench/diagnostic_sidecar.py` | run-bound profile/static/feedback sidecars | diagnostic only; never correctness/timing/score authority | `tests/sol_execbench/core/bench/test_diagnostic_sidecar_statuses.py` |
| Runtime environment evidence | `src/sol_execbench/core/evidence/runtime_evidence/` | host/tool/GPU observations → compatibility report | non-authoritative platform evidence; never a benchmark trace | `tests/sol_execbench/core/evidence/test_runtime_evidence.py` |
| SOLAR public pipeline | `src/solar/api.py` | architecture → extraction → conversion → verification → analysis | exact stage code; no partial publish | `tests/solar/test_api.py` |
| Corpus readiness | `src/sol_execbench/core/solar_bridge/corpus_readiness.py` | pinned scored denominator → three-stage workers → matrix/summary | every workload stays visible; identities and artifacts are hash-bound | `tests/sol_execbench/core/solar_bridge/test_corpus_readiness.py` |
| Graph extraction | `src/solar/graph/extraction.py` | callable trace → typed operator artifact | no einsum converter dependency | SOLAR boundary and API tests |
| Einsum conversion | `src/solar/einsum/conversion.py` | operator artifact → strict semantic graph | exact input/output bindings | SOLAR tests and readability gate |
| Formal analysis | `src/solar/analysis/graph_analyzer.py` | typed analysis job → resource proof | diagnostic results never become scores | SOLAR/Orojenesis tests |
| Scoring formula | `src/sol_execbench/core/scoring/formula.py` | audited runtimes → workload score | no clipping/substitution | `tests/sol_execbench/core/test_sol_score_v3.py` |
| Release planning/execution | `src/sol_execbench/core/scoring/release_builders.py`, `release_runner.py` | pinned corpus → baseline/candidate plans → complete traces | clean source and one immutable environment identity | `tests/sol_execbench/core/scoring/test_release_builders.py` |
| Official verification | `src/sol_execbench/core/scoring/release_verifier.py` | publisher bundle → hash-verified evidence → suite score | raw caller-authored timing JSON is not accepted | `tests/sol_execbench/core/scoring/test_release_verifier.py` |
| Evaluator contract | `src/sol_execbench/core/evaluator_contract.py` | code-owned constants → public machine contract | matches implemented ownership | metadata and score-contract tests |

After moving an entry point or ownership boundary, run Ruff, `ty`,
`scripts/check_coupling.py`, `scripts/check_readability.py` and the focused
tests above.
