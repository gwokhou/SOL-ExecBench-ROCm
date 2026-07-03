# v1.38 Feedback Loop RC1

`v1.38-feedback-loop-rc1` is the SOL-side release tag candidate for
hip-playground consumers that need the diagnostic-only feedback-loop sidecars
introduced after `v1.35`.

This candidate preserves canonical Trace JSONL semantics. It is intended to let
HIP consumers pin a SOL contract that advertises and emits:

- `agent_feedback.sidecar.v1` as `<trace>.agent-feedback.json`
- `profile_summary.sidecar.v1` as `<trace>.profile-summary.json`

Both sidecars are diagnostic-only. They are not correctness, timing, score,
evidence-tier, confirmed-improvement, release-gate, cutover, paper-parity, or
leaderboard authority.

Freshness identity for HIP consumers should prefer `sol_version`,
`candidate_id`, and `source_sha256`. Existing aliases
`sol_contract_version`, `candidate_hash`, and `source_hash` remain documented
and validated for compatibility.

Profiler-derived `summary.bottleneck_hints[]` remain in
`profile_summary.sidecar.v1`. SOL does not duplicate those hints into
`agent_feedback.items[]`; HIP adapters may merge accepted diagnostics after
freshness and authority checks pass for each source sidecar.
