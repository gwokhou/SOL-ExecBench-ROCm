<!-- generated-by: gsd-doc-writer -->
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

## Capabilities

The `capabilities` field is a mapping of current evaluator contract capability
keys to requirement levels. Required trace and baseline semantics use `always`;
optional diagnostics use `optional` or a narrower diagnostic profile.

| Capability key | Level | Meaning |
| --- | --- | --- |
| `trace.correctness` | `always` | Canonical correctness status and correctness metrics. |
| `trace.timing` | `always` | Canonical timing fields and timing interpretation. |
| `trace.scoring` | `always` | Canonical scoring fields and score provenance. |
| `baseline.measured_export` | `always` | Measured baseline registry export fields. |
| `baseline.scoring_artifact` | `always` | Scoring baseline artifact fields. |
| `compatibility.metadata` | `always` | Metadata consumers can persist for compatibility diagnostics. |
| `failure_categories` | `always` | Stable consumer-facing failure buckets. |
| `runtime.evidence` | `optional` | Optional runtime environment evidence beside canonical trace rows. |
| `profiling.evidence` | `optional` | Optional profiler evidence and metadata beside benchmark output. |
| `toolchain.routing` | `optional` | Optional toolchain availability and provenance diagnostics. |
| `static_kernel.evidence` | `optional` | Optional static kernel evidence sidecar diagnostics. |
| `agent_feedback.sidecar` | `profile:diagnostic` | Optional bounded next-experiment feedback diagnostics. |
| `profile_summary.sidecar` | `profile:diagnostic` | Optional normalized profiler summary diagnostics. |

Concrete artifact schema versions are separate from contract capability keys.
`agent_feedback.sidecar` currently emits `sol_execbench.agent_feedback.v2` as
&lt;trace&gt;.agent-feedback.json. `profile_summary.sidecar` currently emits
`sol_execbench.profile_summary.v2` as &lt;trace&gt;.profile-summary.json when a
trace output path is available. Static kernel evidence remains the concrete
`sol_execbench.static_kernel_evidence.v1` sidecar schema behind the
`static_kernel.evidence` capability key. Current ROCm profiling metadata is
still emitted separately as the trace-adjacent &lt;trace&gt;.profile.json
rocprofv3 sidecar.

These capabilities are intentionally optional unless their level is `always`.
A compatible consumer must keep working when a SOL version only provides
canonical trace/profile surfaces and does not produce feedback sidecars. For
HIP freshness checks, SOL sidecars emit canonical identity fields only:
`sol_version`, `candidate_id`, and `source_sha256`.

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
> runner, or sidecar writer invokes the gate. No evaluator capability key is
> advertised for official score evidence by `build_evaluator_contract()` today;
> downstream consumers must treat this artifact as absent. Wiring requires an
> explicit score aggregation policy (not yet a concept on `AmdNativeSuiteReport`)
> and baseline-source classification coverage guarded by tests. Until then,
> AMD-native score reports
> (`sol_execbench.amd_native_score.v1`) remain the only emitted score-adjacent
> surface.

## Feedback Sidecars

`agent_feedback.sidecar`, `profile_summary.sidecar`, and the current
&lt;trace&gt;.profile.json profiler metadata are trace-adjacent diagnostic surfaces.
Their concrete feedback and profile-summary artifact schemas are
`sol_execbench.agent_feedback.v2` and `sol_execbench.profile_summary.v2`.
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
