# Architecture navigation for maintainers

Raw JSON/YAML is validated at parser boundaries. Domain stages consume typed
models or typed request/result objects.

| Domain | Owning entry | Flow | Invariant | Focused checks |
| --- | --- | --- | --- | --- |
| CLI evaluation | `cli/evaluation/evaluator.py` | resolve → package → run → parse → derive/report | staging closes on every path; relative metrics are outer-process work | `tests/sol_execbench/cli/evaluation` |
| Process staging | `driver/problem_packager.py` | normalize → compile assets → stage three runtime templates | execute target is the orchestrator | `tests/sol_execbench/driver/test_problem_packager.py` |
| Trusted reference | `core/bench/reference_service.py` | load reference/input sources → produce case/timing → safetensors response | no candidate import; failures are structured | reference protocol and driver tests |
| Candidate execution | `core/bench/eval_workload_execution.py` | transferred case → correctness → integrity → timing → Trace | no reference load/call; timed outputs validated | `tests/sol_execbench/driver/test_eval_driver.py` |
| Evaluation diagnostics | `core/bench/diagnostic_sidecar.py` | run-bound profile/static/feedback sidecars | diagnostic only; never correctness/timing/score authority | sidecar governance tests |
| Runtime environment evidence | `core/evidence/runtime_evidence/` | host/tool/GPU observations → compatibility report | non-authoritative platform evidence; never a benchmark trace | `core/evidence/test_runtime_evidence.py` |
| SOLAR public pipeline | `solar/api.py` | architecture → extraction → conversion → verification → analysis | exact stage code; no partial publish | `tests/solar/test_api.py` |
| Graph extraction | `solar/graph/extraction.py` | callable trace → typed operator artifact | no einsum converter dependency | SOLAR boundary and API tests |
| Einsum conversion | `solar/einsum/conversion.py` | operator artifact → strict semantic graph | exact input/output bindings | SOLAR tests and readability gate |
| Formal analysis | `solar/analysis/graph_analyzer.py` | typed analysis job → resource proof | diagnostic results never become scores | SOLAR/Orojenesis tests |
| Scoring formula | `core/scoring/formula.py` | audited runtimes → workload score | no clipping/substitution | `core/test_sol_score_v3.py` |
| Official authority | `core/scoring/official_authority.py` | pinned manifest/evidence → fail-closed gate | caller-authored JSON has no authority | `core/dataset/test_aka_corpus.py` |
| Evaluator contract | `core/evaluator_contract.py` | code-owned constants → public machine contract | matches implemented ownership | metadata and score-contract tests |

After moving an entry point or ownership boundary, run Ruff, `ty`,
`scripts/check_coupling.py`, `scripts/check_readability.py` and the focused
tests above.
