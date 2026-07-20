# SOLAR responsibility boundary

The boundary follows SOL-ExecBench §4.2 and is enforced by import tests.

```text
public problems + workload + reference
                 |
                 v
sol_execbench.core.solar_bridge  (the only outer package allowed to import solar)
                 |
                 v
solar.api
  Graph Extractor -> strict extended-einsum converter -> SOL Analyzer
                 |
                 v
hash-bound graphs, conversion proof, formal lower bound
```

`solar` may depend on PyTorch, torchview, graph conversion code, architecture
profiles, and Orojenesis. It must not import `sol_execbench` or model benchmark
definitions, workloads, solutions, candidate timing, baselines, or scores.

`sol_execbench` owns pinned upstream-dataset acquisition and audit, AMD compatibility, seeded
input generation, solution compilation and execution, correctness, candidate
timing, scoring baselines, SOL Score, aggregation, and the user CLI. Production
imports of `solar` are confined to `core/solar_bridge/`, whose worker is isolated
with a timeout, process-group cleanup, and file-backed redacted logs. The
boundary test also scans `tests/sol_execbench/` so test code reaches `solar`
only through the bridge, with one documented exception: the bridge's own
contract tests under `tests/sol_execbench/core/solar_bridge/` may reference
public `solar.api` types (`AnalysisResult`, `AnalysisFailure`, `ArtifactRef`,
`SolBound`) to verify the outcome-mapping logic.

Formal conversion is offline and fail-closed. The converter reads generated
handlers only from `src/solar/handlers/`. The learning command writes candidates
elsewhere and cannot activate them automatically. Formal lookup accepts only
records with passed verification, `formal_review: approved`, matching metadata
and source SHA-256 values, and a safe package-relative source path.

The ROCm formal-publication profile requires a pinned Orojenesis mapper. This is
a stricter release-evidence policy of this port, not an expansion of SOLAR into
benchmark evaluation or a claim of universal paper parity.

The only intended formal target is the packaged `RX_9060_XT` profile and an
observed ROCm `gfx1200` device. Its referenced locked-clock resource-audit file
is not present in this revision, so the profile is explicitly marked
`unavailable` and formal SOLAR publication fails at the architecture stage.
Generic candidate evaluation remains diagnostic; it cannot publish formal
SOLAR artifacts or official scores.
