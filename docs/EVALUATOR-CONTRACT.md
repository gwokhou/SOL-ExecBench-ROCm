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
- `toolchain.routing.v1`
- `static_kernel_evidence.v1`
- `agent_feedback.sidecar.v1`
- `profile_summary.sidecar.v1`

These capabilities are intentionally optional. A compatible consumer must keep
working when a SOL version only provides canonical trace/profile surfaces and
does not produce feedback sidecars.

## Feedback Sidecars

`agent_feedback.sidecar.v1` and `profile_summary.sidecar.v1` are
trace-adjacent diagnostic surfaces. They may guide the next experiment in an
agent loop, but they are not:

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
