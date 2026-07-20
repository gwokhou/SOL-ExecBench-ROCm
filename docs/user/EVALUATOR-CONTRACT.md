# Evaluator Contract v3

The machine-readable contract is emitted by
`sol_execbench.core.evaluator_contract.build_evaluator_contract`. It is the current
ownership map; retired v1/v2 score and baseline surfaces are not authority for
this corpus.

## Ownership

SOL ExecBench owns problem/workload schemas, seeded input generation, isolated
trusted reference preparation, untrusted candidate execution, correctness,
candidate timing, outer-process relative metrics, corpus selection and
materialization, baseline identity, aggregation, and score claims.

SOLAR owns graph extraction, strict extended-einsum conversion, conversion
verification, architecture/resource analysis, and the formal lower-bound
artifact. SOLAR never receives candidate code, candidate latency, baseline
latency, corpus selection, or scores. The only production import from the
outer package into SOLAR is `sol_execbench.core.solar_bridge`.

## Evaluation authority

Canonical Trace JSONL records status, correctness, performance, environment,
clock state, timing protocol, isolation state, and whether timed outputs were
validated. Reference code/output/timing and candidate code/timing live in
distinct processes. Private inherited pipes carry JSON control messages and
safetensors payloads through standard-library `Connection` framing; pickle is
never accepted. The candidate-visible definition contains no trusted reference
source, and the worker-only staged definition is removed before candidate
execution. The ROCm event-timing implementation uses the paper's sampling
counts:

- locked clocks;
- 10 warmup calls per trial;
- 50 timed calls per trial;
- three trials aggregated by arithmetic mean;
- every timed result checked against the reference output;
- serialized access to the selected GPU.

The standard ROCm protocol is labeled
`sol_execbench.rocm_event_timing.paper_counts.v2`; it uses a fixed 256 MiB
cache-clear buffer and a 300-second whole-evaluation timeout. Custom counts,
adaptive duration, or unlocked clocks use
`sol_execbench.rocm_event_timing.custom.v2`. Direct host execution is rejected
unless `--unsafe-local-execution` is explicitly supplied, and such traces are
diagnostic. Performance metadata records the actual adaptive sample count for
every trial in `timed_iterations_per_trial`; `timed_iterations` is populated
only when that count is identical across all trials.

## Formal and score availability

The packaged RX 9060 XT profile references a locked-clock resource audit that
is absent from this revision. It is marked `unavailable`; formal SOLAR analysis
fails at the architecture stage.

The fixed corpus likewise records
`official_scoring.status: unavailable`. No release baseline, independent
rerun, trusted candidate execution attestation, or pinned per-workload SOLAR
manifest set has been published. The current CLI exposes only `score status`,
and no official scorer or baseline generator is implemented. Formula-helper
results and diagnostic speedups are not official scores.

This repository imports a pinned upstream corpus; it does not implement the
paper's dataset extraction/curation pipeline. Candidate static review uses
deterministic AST rules, not the paper's LLM judge. The ROCm formal profile also
requires pinned Orojenesis evidence as a port-specific publication policy.

The implemented formula is:

```text
S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))
```

Incorrect candidates receive zero. Correct inputs require finite positive
runtimes, `T_b > T_SOL`, and `T_k >= T_SOL`; violations are audit failures, not
values to clip or substitute.
