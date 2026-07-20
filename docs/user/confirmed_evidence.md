# Confirmed evidence status

The current v3 release publishes no confirmed official-score artifact. The
only canonical execution artifact is Trace JSONL, whose authority is scoped to
the recorded evaluation. Diagnostic sidecars cannot promote it into a release
score.

`sol-execbench score status` reports unavailable because the pinned corpus has no
published release baseline, independent rerun, trusted candidate attestation,
complete pinned SOLAR manifest set or verified architecture audit.

A future confirmed consumer must validate every item described in
`docs/user/RELEASE-SCORING.md` by immutable identity and checksum. Missing,
duplicate, diagnostic, caller-authored or mismatched inputs remain blockers;
they are never replaced by `reference_latency_ms`, speedup ratios or sidecar
confidence labels.
