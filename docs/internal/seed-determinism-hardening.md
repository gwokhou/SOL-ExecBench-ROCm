# Seed Entropy and Per-Iteration Validation

Status: implemented by the v7 diagnostic/timing transition on 2026-08-02.

## Threats closed

Two evaluation surfaces previously admitted correct-looking near-zero timing:

1. a fixed seed made the timing input reproducible before an evaluation, so a
   candidate could ship a precomputed answer table;
2. pointer-shifted allocations repeated byte-identical values within a trial,
   so a value-keyed cache could compute once and reuse its output.

Static source review remains defense in depth, but the timing protocol no
longer depends on recognizing cache spellings or obfuscation patterns.

## Current protocol

The trusted evaluation boundary generates a fresh 256-bit nonce for every run.
The reference worker retains that nonce; the orchestrator removes it from the
candidate environment before `exec`. Input seeds bind the definition,
workload, row, configured base seed, correctness round, run nonce, trial, and
iteration.

For every warmup and timed invocation:

1. the candidate-side harness requests the next input from the authenticated
   reference worker;
2. the worker generates a fresh input and computes the matching reference
   output, retaining the output in private process memory;
3. only the input is sent to the candidate process;
4. the candidate invocation is timed, and CUDA outputs must already reside on
   the selected GPU before the end event;
5. the actual output is sent back to the worker for one-shot shape, dtype, and
   numerical validation;
6. the worker returns only success or failure, then consumes the pending case.

Requesting another input before validating the current case fails closed.
Repeated input-content hashes within a trial also fail closed. Consequently, a
custom generator that ignores its seeded RNG, an all-constant workload, or an
all-fixed safetensors workload is not eligible for the v4 timing protocol until
it provides a seed-sensitive input route.

The standard sampling counts remain ten warmups, fifty timed iterations, and
three trials. Setup, IPC, and trusted reference execution occur outside device
events; only the candidate invocation is measured. Every warmup and timed
result is nevertheless validated so the reference-worker state machine cannot
be skipped.

## Authority and isolation

Reference source, expected outputs, input nonce, and validation state never
enter candidate process memory. The generated driver still invokes candidate
code in its own evaluator process, so sealed timing-function and call-graph
identity checks remain required. Publication runs additionally require the
container boundary: no network, all capabilities dropped, Docker's default
seccomp profile, `no-new-privileges`, private IPC, an 8 GiB shared-memory
allocation, and a 512-PID ceiling.

This is not a claim that static review recognizes every possible hostile
program. Instead, official timing no longer exposes the reference output that
the known compute-once and call-stack attacks need, and any protocol or
integrity violation invalidates the trace.

## Artifact transition

The change is intentionally breaking. The canonical identifiers are now:

- `sol_execbench.benchmark_config.v2`;
- `sol_execbench.reference_ipc.v2`;
- `sol_execbench.rocm_event_timing.paper_counts.v4`;
- `sol_execbench.rocm_event_timing.custom.v4`;
- `sol_execbench.performance_diagnostic.v7` and its current evidence,
  inference, acceptance, and feedback family versions.

No compatibility readers or migrations accept superseded timing evidence.
Calibration, the 440-case development corpus, the 220-case held-out corpus,
inference, acceptance, and governed Agent feedback were recollected and frozen
under these identifiers. The resulting v7 statistical acceptance is rejected,
so no code-changing feedback action is enabled.
