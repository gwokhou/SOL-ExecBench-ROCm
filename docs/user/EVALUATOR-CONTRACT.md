# Evaluator Contract v3

The machine-readable contract is emitted by
`sol_execbench.core.evaluator_contract.build_evaluator_contract`. It is the current
ownership map for this corpus.

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
`sol_execbench.rocm_event_timing.paper_counts.v3`; it uses a cache-clear buffer
equal to twice the detected L2 size (256 MiB only when L2 detection is unavailable)
and a 300-second whole-evaluation timeout. Custom counts,
adaptive duration, or unlocked clocks use
`sol_execbench.rocm_event_timing.custom.v3`. Direct host execution is rejected
unless `--unsafe-local-execution` is explicitly supplied, and such traces are
diagnostic. Performance metadata records the actual adaptive sample count for
every trial in `timed_iterations_per_trial`; `timed_iterations` is populated
only when that count is identical across all trials.

During the timed region, standard Python thread-start entry points are guarded
synchronously, so a worker that starts and exits between count samples is still
rejected. Concurrent thread-count sampling remains defense in depth for
alternate entry points, while static review blocks direct `threading`,
`_thread`, multiprocessing, concurrent-executor, non-default stream, graph
capture, and TorchScript-fork sources.

## Formal and score availability

The packaged RX 9060 XT profile references a content-addressed locked-clock v3
resource audit. Formal SOLAR analysis verifies its exact non-exempt
precision/resource coverage and all required ISA/code-object/runtime instruction
checks, together with its frozen tuning and held-out measurement evidence,
before graph extraction.

The fixed corpus likewise records
`official_scoring.status: unavailable`. No release baseline, independent
rerun, trusted candidate execution attestation, or pinned per-workload SOLAR
manifest set has been published. The CLI implements a deterministic
trusted-reference baseline generator and a four-authority signed-bundle verifier,
but neither can self-publish trust roots or upgrade unsigned local evidence.
Formula-helper results and diagnostic speedups are not official scores.
Release execution additionally requires a clean source tree at the declared Git
revision, records the immutable Docker image ID, and rejects byte-for-byte reuse
of baseline traces as an independent rerun.

This repository imports a pinned upstream corpus; it does not implement the
paper's dataset extraction/curation pipeline. Candidate static review uses
deterministic AST rules, not the paper's LLM judge. The ROCm formal profile also
requires pinned Orojenesis evidence as a port-specific publication policy.
The formal CLI sets that requirement unconditionally, and the worker, bridge,
and CLI reject diagnostic bounds independently. A reviewed mapper binary and
its pinned provenance/build manifest are both required; the current empty
binary allowlist keeps publication unavailable.

The implemented formula is:

```text
S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))
```

Incorrect candidates receive zero. Correct inputs require finite positive
runtimes, `T_b > T_SOL`, and `T_k >= T_SOL`; violations are audit failures, not
values to clip or substitute.
