# hip-execbench Practice Map

This note records which engineering practices from a separate hip-execbench
project are worth adapting into SOL ExecBench ROCm. It is intentionally an
adaptation map, not a porting plan:
public `definition.json`, `solution.json`, workload formats, CLI behavior,
trace JSONL output, and benchmark semantics remain owned by SOL ExecBench ROCm.

Source evidence reviewed from hip-execbench included `src/profiler/router.ts`,
`src/errors/index.ts`, `src/agent/builder.ts`, `src/baseline/comparator.ts`,
`src/schemas/*.ts`, and `src/pipeline/statistics.ts`.

## Accepted Practices

| Practice | hip-execbench source evidence | SOL ExecBench ROCm adaptation | Public contract impact |
|----------|-------------------------------|-------------------------------|------------------------|
| Tool readiness diagnostics | Backend, reason, fallback status, and effective profiling-level reporting. | Add internal ROCm diagnostic helpers that report tool presence and selected profiling readiness without adding CLI flags. | None |
| Structured failure surfaces | Typed errors carrying path, field, expected, actual, detail, and fix-hint context. | Add internal stage/error helpers for parse, packaging, compile, runtime, verification, timing, and environment failures. | None |
| Agent/report transformation layer | Stable document generation through pure transformation helpers. | Add pure derived trace summary helpers for local reporting and tests; do not add fields to trace JSONL. | None |
| Baseline/scoring discipline | Named-source comparison with thresholds and explicit repeated-run statistics boundaries. | Keep baseline comparison baseline-relative and warn against unsupported AMD performance claims. | None |
| Trace-file baseline comparison | Named baseline comparison and summary emission. | Keep `sol-execbench-baseline` over existing trace JSONL with WIN/PARITY/LOSS classifications and JSON/text output. | Additive CLI only; existing `sol-execbench` behavior and trace schema unchanged |

## Rejected Practices

| Practice | Reason |
|----------|--------|
| Replacing the Python CLI with alternate subcommands | A distinct command shape would change public CLI contracts and duplicate existing SOL ExecBench entry points. |
| Switching schemas to a TypeScript/Zod model | Importing that model would change the public Pydantic schema surface and validation behavior. |
| Emitting a new agent JSON contract from normal runs | A transformation pattern can be useful, but trace JSONL is the public machine-readable contract in this project. |
| Importing HTML/Plotly reports | Frontend/reporting dependencies would add a new report surface outside this milestone. |
| Replacing trace JSONL with alternate trace blocks | A separate trace model would change the established SOL ExecBench trace contract. |

## Deferred Practices

| Practice | Reason |
|----------|--------|
| Importing Mann-Whitney U significance tests directly | SOL ExecBench ROCm currently records one trace per workload by default; significance testing needs an explicit repeated-sample contract before it is meaningful. |
| Hardware/tool caching as an optimization | Expensive probe caching should be deferred until diagnostics probes are implemented and measured. |
| Claiming CDNA 3 hardware validation | User scope is CDNA 3 implementation/readiness only; real `gfx94*` hardware validation remains a future milestone. |

## Guardrails

- Adaptations must be internal helpers, tests, or documentation unless a later
  milestone explicitly approves a public contract change.
- Existing benchmark definition, workload, solution, and trace JSONL formats
  are compatibility contracts.
- SOL-Score-style output must not be presented as an AMD-native hardware claim
  until a dedicated AMD interpretation model is defined and validated.
- Baseline comparison is baseline-relative by default. It may be used for local
  regression or replacement comparison, but not as standalone AMD roofline
  evidence.
- CDNA 3 remains schema/build/docs-supported but not hardware-validated until a
  real full-suite `gfx94*` run is recorded.
