# Agent Feedback Sidecar

`sol_execbench.agent_feedback.v3` is an optional diagnostic sidecar written next
to canonical Trace JSONL by `_agent_feedback_sidecar_path` in
`src/sol_execbench/cli/sidecars/agent_feedback.py`. It gives downstream agents
bounded next-experiment guidance while preserving Trace JSONL as the only
authority for correctness, timing, scoring, and evaluation status.

The evaluator contract advertises this artifact through the optional
`agent_feedback.sidecar` capability key.

The sidecar is not score authority, evidence-tier authority, confirmed
improvement authority, release-gate authority, cutover authority, paper-parity
authority, leaderboard authority, or claim-upgrade authority. Missing,
malformed, contradictory-authority, stale, partial, unavailable, or unknown
feedback is a diagnostic state only.

## HIP Consumer Mapping

HIP Playground should treat the SOL sidecar as an adapter input, not as a
benchmark result. Suggested mapping:

| SOL field | HIP use | Safe fallback |
| --- | --- | --- |
| `status` | Availability class for feedback ingestion. | Treat unknown values as unavailable. |
| `reason_code` | Stable reason for generation state. | Preserve as opaque diagnostic text or downgrade to unknown. |
| `items[].bottleneck` | Candidate strategy hint taxonomy. | Unknown values must be downgraded to an `unknown` bucket. |
| `items[].recommendation` | Prompt-safe next-experiment instruction. | Omit if absent or if source freshness is stale. |
| `limitations[]` | Guardrails to include beside any agent hint. | Always preserve diagnostic-only wording. |
| `source_refs[]` | Compact evidence source categories. | Ignore unsupported source kinds. |
| `artifact_citations[]` | Compact path and checksum references for trace-adjacent artifacts. | Reject absolute paths and missing checksums where a checksum is required. |
| `identity` | Freshness denominator for trace/run/candidate matching. | Reject as stale when expected identity mismatches. |
| `authority` | Hard guardrail enum. | Reject unsupported or non-diagnostic authority values. |

SOL emits a closed `items[].bottleneck` vocabulary:

- `unknown`
- `compile_failure`
- `runtime_failure`
- `timeout`
- `numerical_correctness`
- `interface_correctness`
- `policy_violation`
- `reference_failure`

Schema validation rejects other SOL-produced bottleneck labels. HIP adapters
should still downgrade unrecognized future values to `unknown` at their boundary
instead of promoting them into a prompt taxonomy.

The CLI fills `identity.target_id`, `identity.run_id`, `identity.candidate_id`,
`identity.source_sha256`, and `identity.sol_version` from emitted trace data
when available. `run_id` is the persisted Trace JSONL checksum when the trace
file exists. `candidate_id` is a checksum of the compact solution labels in
emitted trace rows, not a source-content identity. When the CLI has loaded the
`Solution`, `identity.source_sha256` is the solution content hash covering
source paths, source contents, build metadata, and dependencies. Consumers that
need strong candidate-source freshness should prefer `source_sha256` over
`candidate_id`.

HIP runtime prompts should never include raw trace rows, raw profiler dumps, full
kernel source, prompt text, or absolute temporary paths from SOL feedback. They
should include only the normalized bottleneck, recommendation, limitation, and
compact citation fields after freshness and authority checks pass.

The `profile_summary.sidecar` evaluator capability key advertises the concrete
`sol_execbench.profile_summary.v3` normalized profile-summary sidecar for
bounded profiler metrics, conservative bottleneck hints, and artifact
citations. Current ROCm profiler metadata remains the separate rocprofv3
sidecar written by `_profile_sidecar_path` in
`src/sol_execbench/cli/sidecars/profile.py` and is cited as optional diagnostic
evidence when present. Profiler-derived `summary.bottleneck_hints[]` remain in
the profile-summary contract; SOL does not duplicate those hints into
`agent_feedback.items[]`. HIP adapters may combine accepted profile-summary
hints with accepted agent-feedback items after both sidecars pass freshness and
authority checks. Agent feedback and profile summary remain separate diagnostic
surfaces; neither is score, release-gate, or cutover authority.

For all closed HIP taxonomies, unknown values must be downgraded rather than
promoted.

## Fixture Package

CPU-safe fixtures live under
`tests/sol_execbench/fixtures/agent_feedback/`.

| Fixture | Purpose | Expected HIP handling |
| --- | --- | --- |
| `tests/sol_execbench/fixtures/agent_feedback/valid.agent-feedback.json` | Current, available sidecar with passing trace guidance. | Accept as diagnostic next-experiment guidance. |
| `tests/sol_execbench/fixtures/agent_feedback/partial.agent-feedback.json` | Partial diagnostics because profile evidence is unavailable. | Accept with limitation text and lower confidence. |
| `tests/sol_execbench/fixtures/agent_feedback/unavailable.agent-feedback.json` | No evaluated traces were available. | Treat as unavailable diagnostic state. |
| `tests/sol_execbench/fixtures/agent_feedback/stale.agent-feedback.json` | Valid schema with old trace/run/candidate identity. | Reject as stale for the current run. |
| `tests/sol_execbench/fixtures/agent_feedback/malformed.agent-feedback.json` | Invalid JSON payload. | Treat as invalid diagnostic state. |
| `tests/sol_execbench/fixtures/agent_feedback/contradictory-authority.agent-feedback.json` | Schema-shaped payload with forbidden score authority enum. | Reject before prompt assembly. |
| `tests/sol_execbench/fixtures/agent_feedback/missing.agent-feedback.case.json` | Metadata fixture for absent sidecar path behavior. | Treat missing sidecar as unavailable. |

These fixtures use synthetic checksums and compact file names. They are not
benchmark evidence and do not represent real profiler output.

## Consumer Rules

1. Parse with strict schema validation.
2. Reject malformed JSON and contradictory authority payloads before prompt
   assembly.
3. Validate freshness against the current trace path and any available
   target/run/candidate/source identity.
4. Downgrade stale, unknown, missing, unavailable, and partial states instead of
   promoting benchmark claims.
5. Map `bottleneck`, `recommendation`, `limitation`, and `artifact_citations`
   into a closed HIP taxonomy. Unknown values must be downgraded, not promoted.
6. Keep canonical Trace JSONL as the only source for correctness, timing,
   scoring, and status.
