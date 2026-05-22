# hip-execbench Practice Map

This note records which engineering practices from
`~/PyCharmMiscProject/hip-playground/hip-execbench` are worth adapting into
SOL ExecBench ROCm. It is intentionally an adaptation map, not a porting plan:
public problem formats, solution formats, CLI behavior, trace JSONL output, and
benchmark semantics remain owned by SOL ExecBench ROCm.

## Accepted Practices

| Practice | Source pattern | SOL ExecBench ROCm adaptation | Public contract impact |
|----------|----------------|-------------------------------|------------------------|
| Tool readiness diagnostics | Profiler router and PMC availability checks distinguish basic/full profiling and fallback reasons. | Add internal ROCm diagnostic helpers that report tool presence and selected profiling readiness without adding CLI flags. | None |
| Structured failure surfaces | Typed errors carry stage and actionable fix hints. | Add internal stage/error helpers for parse, packaging, compile, runtime, verification, timing, and environment failures. | None |
| Agent/report transformation layer | Agent builder turns pipeline internals into stable machine-readable summaries. | Add pure trace summary helpers for local reporting and tests; do not add fields to trace JSONL. | None |
| Baseline/scoring discipline | Baseline comparison uses explicit thresholds and significance tests. | Add documentation and tests that keep SOL-Score-style output stable and warn against unsupported AMD performance claims. | None |
| Trace-file baseline comparison | `src/baseline/` compares agent results with named baselines and emits markdown summaries. | Add `sol-execbench-baseline` over existing trace JSONL with WIN/PARITY/LOSS classifications and JSON/text output. | Additive CLI only; existing `sol-execbench` behavior and trace schema unchanged |
| Hardware/tool caching as an optimization | PMC counter discovery caches expensive subprocess calls. | Keep as a future optimization candidate for diagnostics if profiling probes become expensive. | None |

## Rejected Or Deferred Practices

| Practice | Reason |
|----------|--------|
| Replacing the Python CLI with `hip-bench` subcommands | Would change public CLI contracts and duplicate existing SOL ExecBench entry points. |
| Switching schemas to the TypeScript/Zod model | Would change the public Pydantic schema surface and validation behavior. |
| Emitting a new agent JSON contract from normal runs | Trace JSONL is the public machine-readable contract in this project. |
| Importing `hip-execbench` Mann-Whitney U significance tests directly | SOL ExecBench ROCm currently records one trace per workload by default; significance testing needs an explicit repeated-sample contract before it is meaningful. |
| Importing `hip-execbench` HTML/Plotly reports | Adds frontend/reporting dependencies and a new report surface beyond the v1.3 closure goal. |
| Replacing trace JSONL with `hip-execbench` trace blocks | Would change the established SOL ExecBench trace contract. |
| Claiming CDNA 3 hardware validation | User explicitly deferred real `gfx94*` validation for this milestone. |

## Guardrails

- Adaptations must be internal helpers, tests, or documentation unless a later
  milestone explicitly approves a public contract change.
- Existing `definition.json`, `workload.jsonl`, `solution.json`, and trace JSONL
  formats are compatibility contracts.
- SOL-Score-style output must not be presented as an AMD-native hardware claim
  until a dedicated AMD interpretation model is defined and validated.
- Baseline comparison is baseline-relative by default. It may be used for local
  regression or replacement comparison, but not as standalone AMD roofline
  evidence.
- CDNA 3 remains schema/build/docs-supported but not hardware-validated until a
  real full-suite `gfx94*` run is recorded.
