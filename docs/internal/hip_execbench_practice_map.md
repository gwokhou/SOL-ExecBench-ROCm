# hip-execbench Practice Map

This note records which engineering practices from
`~/PyCharmMiscProject/hip-playground/hip-execbench` are worth adapting into
SOL ExecBench ROCm. It is intentionally an adaptation map, not a porting plan:
public problem formats, solution formats, CLI behavior, trace JSONL output, and
benchmark semantics remain owned by SOL ExecBench ROCm.

## Accepted Practices

| Practice | hip-execbench source evidence | SOL ExecBench ROCm adaptation | Public contract impact |
|----------|-------------------------------|-------------------------------|------------------------|
| Tool readiness diagnostics | `src/profiler/router.ts` returns backend, reason, fallback status, and effective profiling level for RDNA/CDNA/tool combinations. | Add internal ROCm diagnostic helpers that report tool presence and selected profiling readiness without adding CLI flags. | None |
| Structured failure surfaces | `src/errors/index.ts` defines typed errors carrying path, field, expected, actual, detail, and fix-hint context. | Add internal stage/error helpers for parse, packaging, compile, runtime, verification, timing, and environment failures. | None |
| Agent/report transformation layer | `src/agent/builder.ts` maps pipeline data into a stable agent document through pure transformation helpers. | Add pure derived trace summary helpers for local reporting and tests; do not add fields to trace JSONL. | None |
| Baseline/scoring discipline | `src/baseline/comparator.ts` compares named sources with thresholds and repeated-run pairwise tests. | Keep baseline comparison baseline-relative and warn against unsupported AMD performance claims. | None |
| Trace-file baseline comparison | `src/baseline/comparator.ts` and `src/baseline/comparisonReport.ts` compare named baselines and emit summaries. | Keep `sol-execbench-baseline` over existing trace JSONL with WIN/PARITY/LOSS classifications and JSON/text output. | Additive CLI only; existing `sol-execbench` behavior and trace schema unchanged |

## Rejected Practices

| Practice | Reason |
|----------|--------|
| Replacing the Python CLI with `hip-bench` subcommands | `src/cli/index.ts` and `src/main.ts` implement a distinct command shape that would change public CLI contracts and duplicate existing SOL ExecBench entry points. |
| Switching schemas to the TypeScript/Zod model | `src/schemas/*.ts` uses Zod; importing that model would change the public Pydantic schema surface and validation behavior. |
| Emitting a new agent JSON contract from normal runs | `src/agent/builder.ts` is useful as a transformation pattern, but trace JSONL is the public machine-readable contract in this project. |
| Importing `hip-execbench` HTML/Plotly reports | `src/reporting/html.ts` and `src/reporting/charts.ts` add frontend/reporting dependencies and a new report surface outside this milestone. |
| Replacing trace JSONL with `hip-execbench` trace blocks | `src/tracing/trace.ts` is a separate trace model and would change the established SOL ExecBench trace contract. |

## Deferred Practices

| Practice | Reason |
|----------|--------|
| Importing `hip-execbench` Mann-Whitney U significance tests directly | `src/pipeline/statistics.ts` is a solid pure implementation, but SOL ExecBench ROCm currently records one trace per workload by default; significance testing needs an explicit repeated-sample contract before it is meaningful. |
| Hardware/tool caching as an optimization | `src/profiler/pmc-cache.ts` and `src/profiler/profile-cache.ts` cache expensive probes, but SOL ExecBench ROCm should defer this until diagnostics probes are implemented and measured. |
| Claiming CDNA 3 hardware validation | User scope is CDNA 3 implementation/readiness only; real `gfx94*` hardware validation remains a future milestone. |

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
