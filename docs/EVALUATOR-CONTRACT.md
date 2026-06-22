# Evaluator Contract

`sol-execbench contract --json` prints the GPU-free compatibility contract that
downstream tools can inspect before running a benchmark. The contract is
metadata about SOL-owned behavior. It is not itself a Trace row and it does not
extend the canonical Trace JSONL schema.

## Canonical Authority

SOL owns the benchmark truth for:

- correctness status and correctness metrics
- timing fields and timing interpretation
- score fields and score provenance
- evaluation status vocabulary
- canonical Trace JSONL field groups

Downstream consumers may persist contract metadata for compatibility
diagnostics, but they must not redefine SOL benchmark truth.

## Optional Capabilities

Optional capability tokens advertise sidecar or diagnostic surfaces that may be
available beside a run:

- `runtime.evidence.v1`
- `profiling.evidence.v1`
- `official_score_evidence.v1`
- `toolchain.routing.v1`
- `static_kernel_evidence.v1`
- `agent_feedback.sidecar.v1`
- `profile_summary.sidecar.v1`

These capabilities are intentionally optional. A compatible consumer must keep
working when a SOL version only provides canonical trace/profile surfaces and
does not produce feedback sidecars.

`agent_feedback.sidecar.v1` is the concrete optional JSON sidecar written as
`<trace>.agent-feedback.json`. `profile_summary.sidecar.v1` is the concrete
optional normalized profile summary written as `<trace>.profile-summary.json`
when a trace output path is available. Current ROCm profiling metadata is still
emitted separately as the trace-adjacent `<trace>.profile.json` rocprofv3
sidecar. Consumers must keep working when either optional sidecar is absent.

## Official Score Evidence

`official_score_evidence.v1` is the SOL-owned score-authoritative evidence
surface for confirmed benchmark score claims. It is separate from
`sol_execbench.amd_native_score.v1`: AMD-native score reports remain derived
inputs and may be cited by official evidence, but they are not by themselves
confirmed official score artifacts.

A workload official score is non-null only when the evidence gate has all
required inputs:

- measured candidate latency
- official measured baseline latency
- SOL/SOLAR bound evidence
- explicit aggregation policy

If any required input is absent, the official score value is `null` and the
report carries stable `blocker_reason_codes`, including `missing_score`,
`missing_baseline`, `placeholder_baseline`, `missing_sol_bound`,
`missing_measured_latency`, and `missing_aggregation_policy`.

Trace JSONL `evaluation.performance.speedup_factor` remains a diagnostic ratio
of reference latency to candidate latency. It is not an official benchmark
score and must not be substituted for `official_score_evidence.v1` score
fields.

> **Integration status (staging).** The gating logic and data models
> (`sol_execbench.official_score_evidence.v1`) are delivered in
> `src/sol_execbench/core/scoring/official_score.py` and re-exported from the
> scoring package, but **no run path emits this artifact yet** — no CLI command,
> runner, or sidecar writer invokes the gate. The `official_score_evidence.v1`
> capability token is advertised as a future surface; downstream consumers must
> treat it as absent today. Wiring requires an explicit score aggregation policy
> (not yet a concept on `AmdNativeSuiteReport`) and baseline-source
> classification coverage guarded by tests. Until then, AMD-native score reports
> (`sol_execbench.amd_native_score.v1`) remain the only emitted score-adjacent
> surface.

## Feedback Sidecars

`agent_feedback.sidecar.v1`, `profile_summary.sidecar.v1`, and the current
`<trace>.profile.json` profiler metadata are trace-adjacent diagnostic surfaces.
They may guide the next experiment in an agent loop, but they are not:

- correctness authority
- performance or timing authority
- score authority
- evidence-tier authority
- confirmed-improvement authority
- release-gate authority
- cutover authority
- paper-parity authority
- leaderboard authority

The canonical Trace JSONL remains the only compatibility surface for benchmark
status, correctness, timing, and scoring semantics.

## Ownership Boundary

SOL owns feedback sidecar schema, generation, freshness identity, artifact
citations, and authority guardrails. HIP consumers own adapter normalization,
closed-taxonomy `ProfileDigest` mapping, strategy hints, and prompt assembly.

This split keeps optional feedback useful for next-turn diagnostics without
making prompt-facing summaries part of the benchmark authority model.
