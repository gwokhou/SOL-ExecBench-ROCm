# Large batch GPU qualification

Every repository-owned large batch GPU producer uses the same mandatory stage
names and ordering:

```text
qualify-static -> qualify-canary -> qualify-full -> run/collect/release-build
```

`qualify-static` validates the complete immutable input denominator without
using the GPU. `qualify-canary` runs per-axis extrema in risk-first order.
`qualify-full` executes every item with a minimal correctness-oriented protocol.
Qualification timing is always `performance_authority=false`; it cannot become
score, calibration, acceptance, or publication evidence.

Every stage writes a content-bound gate and receipts below an isolated
`--qualification-root`. The next stage verifies its immediate parent. The
formal producer verifies the complete chain before its first formal workload,
profiler pass, mapper run, or repeated calibration batch. Subject, runner,
configuration, source, selected items, evidence hashes, and hardware evidence
must still match. A partial directory or exit code without receipts is not a
gate.

## Governed task inventory

| Large batch task | Uniform qualification entrypoint | Gated producer |
| --- | --- | --- |
| Diagnostic counter collection | `build_rdna4_diagnostic_corpora.py qualify-*` | `collect` |
| Baseline/candidate release evaluation | `sol-execbench baseline qualify-*` | `baseline release-run` |
| Full-corpus SOLAR release | `sol-execbench solar qualify-*` | `solar release-build` |
| AKA tolerance calibration | `aka_calibrate_tolerances.py qualify-*` | `run` |
| SOLAR cross-path focus | `run_cross_path_focus.py qualify-*` | `run` |
| RDNA4 diagnostic calibration | `run_rdna4_diagnostic_calibration.py qualify-*` | `run` |
| RDNA4 resource-peak calibration | `run_qualified_rdna4_resource_peak_calibration.py qualify-*` | `run` |

The resource-peak v3 evidence pins the complete SHA-256 of
`run_rdna4_resource_peak_calibration.py`. That historical producer is retained
byte-for-byte so the real audit remains verifiable. New executions must use the
qualified successor launcher; changing the pinned producer or rewriting its
historical digest would invalidate evidence.

Single-workload `evaluate` and bounded profiler-overhead probes are not large
batch producers. The RDNA4 hardware test bundle is itself a validation job, not
a producer of formal measurements. Diagnostic lifecycle `run` adopts and
verifies already-produced artifacts; it does not launch GPU collection. These
paths therefore do not receive a synthetic qualification gate.

## Release evaluation

Run static qualification first, then run canary and full qualification inside
the same hardened container used for the release:

```bash
sol-execbench baseline qualify-static PLAN \
  --qualification-root QUALIFICATION_ROOT
./scripts/run_docker.sh -- sol-execbench baseline qualify-canary PLAN \
  --qualification-root QUALIFICATION_ROOT
./scripts/run_docker.sh -- sol-execbench baseline qualify-full PLAN \
  --qualification-root QUALIFICATION_ROOT
./scripts/run_docker.sh -- sol-execbench baseline release-run PLAN \
  --qualification-root QUALIFICATION_ROOT
```

The canary selects every axis minimum and maximum within every scored problem.
Full qualification covers the exact release denominator with zero warmups, one
iteration, one trial, no profiler, and no clock-lock requirement. Formal
release timing remains separate and unchanged.

## SOLAR release

The former optional `solar corpus-audit` command is replaced by the mandatory
uniform chain:

```bash
sol-execbench solar qualify-static RELEASE_WORKSPACE \
  --orojenesis-home OROJENESIS_HOME \
  --qualification-root QUALIFICATION_ROOT
sol-execbench solar qualify-canary RELEASE_WORKSPACE \
  --orojenesis-home OROJENESIS_HOME \
  --qualification-root QUALIFICATION_ROOT
sol-execbench solar qualify-full RELEASE_WORKSPACE \
  --orojenesis-home OROJENESIS_HOME \
  --qualification-root QUALIFICATION_ROOT
sol-execbench solar release-build RELEASE_WORKSPACE \
  --orojenesis-home OROJENESIS_HOME \
  --qualification-root QUALIFICATION_ROOT
```

Canary and full qualification run the isolated extraction, strict conversion,
and replay checks. The full readiness matrix and summary are gate evidence,
not formal SOLAR output. Backend, device, timeout, Orojenesis path, jobs policy,
corpus identity, architecture identity, and source revision are bound.

## Internal calibration and focused analysis

The internal runners use the same positional stage and require the same
arguments at every stage so the intended formal configuration is bound:

```bash
uv run python SCRIPT qualify-static --qualification-root QUAL_ROOT [ARGS]
uv run python SCRIPT qualify-canary --qualification-root QUAL_ROOT [ARGS]
uv run python SCRIPT qualify-full --qualification-root QUAL_ROOT [ARGS]
uv run python SCRIPT run --qualification-root QUAL_ROOT [ARGS]
```

Fixed-probe calibration canaries cover representative compute, memory,
reduction/atomic, and overlap probes. Full qualification minimally executes
every fixed probe once before tuning and multi-process estimation. AKA and
cross-path canaries use workload-axis extrema; their full gates cover their
complete executable denominators.
