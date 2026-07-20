# Profile Summary Sidecar

`sol_execbench.profile_summary.v3` is an optional diagnostic sidecar written
next to canonical Trace JSONL using the profile-summary JSON suffix. It normalizes
bounded `rocprofv3` profile metadata for downstream adapters while preserving
Trace JSONL as the only authority for correctness, timing, scoring, and
evaluation status.

The evaluator contract advertises this artifact through the optional
`profile_summary.sidecar` capability key. The concrete artifact schema remains
`sol_execbench.profile_summary.v3`.

The summary sidecar does not replace the raw profile JSON sidecar. The existing
profile metadata sidecar remains the raw diagnostic metadata record and is cited
from the normalized profile summary when present.

When a `rocprofv3` profile result is supplied, the summary carries bounded
artifact registration metadata from `sol_execbench.rocprofv3_profile.v1`:
`summary.artifact_coverage_status`, `summary.reason_codes`,
`summary.warnings`, `summary.artifact_count`, and `summary.artifact_kinds`.
Return-code-zero profile runs can report partial profiler status when
artifact coverage is incomplete or non-collectable. Partial or missing coverage
is represented with stable reason codes such as
`rocprof_no_registered_artifacts` and `rocprof_partial_artifact_coverage`.

The summary may also include structured diagnostic evidence:

- `summary.workload_metrics[]`: workload-level bounded metrics such as artifact
  coverage and dispatch counts.
- `summary.kernel_metrics[]`: kernel-level bounded metrics derived from
  registered CSV/JSON profiler artifacts, such as kernel duration rows and
  selected numeric counters.
- `summary.bottleneck_hints[]`: conservative AMD-oriented diagnostic hints.
- `summary.parse_warnings[]`: bounded parse warnings for malformed,
  unsupported, too-large, or citation-only artifacts.

Phase 191 parses only registered CSV and JSON text artifacts. `.rocpd`,
database, Perfetto/PFTrace, and OTF2 artifacts remain citation-only evidence
pointers in this version.

The sidecar is not correctness authority, timing authority, performance
authority, score authority, evidence-tier authority, confirmed-improvement
authority, release-gate authority, cutover authority, paper-parity authority,
leaderboard authority, or claim-upgrade authority. Missing, malformed,
contradictory-authority, stale, partial, unavailable, or unknown summaries are
diagnostic states only.

## HIP Consumer Mapping

HIP Playground should treat the SOL profile summary as optional adapter input,
not as a benchmark result. Suggested mapping:

| SOL field | HIP use | Safe fallback |
| --- | --- | --- |
| `status` | Availability class for normalized profile metadata. | Treat unknown values as unavailable. |
| `reason_code` | Stable generation state. | Preserve as opaque diagnostic text or downgrade to unknown. |
| `identity` | Freshness denominator for trace/run matching. `identity.sol_version` is the canonical SOL producer version. | Reject as stale when expected identity mismatches. |
| `summary.profiler_status` | Raw rocprofv3 collection status. | Treat absent status as unavailable. |
| `summary.artifact_coverage_status` | Registration coverage class such as complete, partial, none, or unavailable. | Treat unknown values as partial diagnostics. |
| `summary.reason_codes` | Stable rocprofv3 registration and availability reason codes. | Preserve unknown codes as opaque diagnostic text. |
| `summary.warnings` | Bounded profile artifact coverage warnings. | Display as advisory text only. |
| `summary.artifact_count` | Coarse evidence richness signal. | Use zero when absent. |
| `summary.artifact_kinds` | Compact profile artifact inventory. | Ignore unsupported kinds. |
| `summary.metrics[]` | Bounded metadata-derived metrics. | Ignore unknown metric names. |
| `summary.workload_metrics[]` | Workload-level diagnostic metrics with source artifact and parse status. | Ignore unknown metric names or unsupported parse status values. |
| `summary.kernel_metrics[]` | Kernel-level diagnostic metrics from bounded CSV/JSON artifacts. | Ignore unknown metric names and never treat them as official timing. |
| `summary.bottleneck_hints[]` | Conservative diagnostic hint taxonomy. | Downgrade unknown categories to `unknown`. |
| `summary.parse_warnings[]` | Bounded warnings explaining skipped or partial parsing. | Display as advisory text only. |
| `limitations[]` | Guardrails to include beside any profile digest. | Always preserve diagnostic-only wording. |
| `artifact_citations[]` | Compact path, size, status, and checksum references for trace, raw profile metadata, and profiler artifacts. | Reject absolute paths and missing checksums where a checksum is required. |
| `authority` | Hard guardrail enum. | Reject unsupported or non-diagnostic authority values. |

`summary.bottleneck_hints[]` uses this closed vocabulary:

- `compute_bound`
- `memory_l2_bound`
- `lds_bound`
- `launch_overhead`
- `insufficient_counters`
- `unknown`

These hints are diagnostic adapter input only. They are conservative and do not
claim fine-grained occupancy, VGPR/SGPR pressure, cache, or bandwidth
conclusions. When counters are missing or insufficient, SOL emits
`insufficient_counters` or `unknown` instead of speculating.

Profiler-derived bottleneck hints remain in the
`sol_execbench.profile_summary.v3` sidecar; SOL does not surface them as
`agent_feedback.items[]`. HIP adapters that want one prompt-facing hint list
should merge profile-summary hints with agent-feedback items only after each
source sidecar passes strict freshness and authority checks.

HIP runtime prompts should never include raw trace rows, raw profiler dumps,
full kernel source, prompt text, or absolute temporary paths from SOL profile
summaries. They should include only normalized status, bounded metrics,
limitations, and compact citations after freshness and authority checks pass.
Profiler artifact citations use compact file names and compute SHA256 by
default. Hashing large `.rocpd` or database artifacts can be expensive; SOL
currently prefers auditability over skipping hashes, and any future size limit
should be treated as a contract change rather than inferred by HIP.

For all closed HIP taxonomies, unknown values must be downgraded rather than
promoted.

## Fixture Package

CPU-safe fixtures live under
`tests/sol_execbench/fixtures/profile_summary/`.

| Fixture | Purpose | Expected HIP handling |
| --- | --- | --- |
| `tests/sol_execbench/fixtures/profile_summary/valid.profile-summary.json` | Current, available sidecar with profile metadata and artifact citations. | Accept as diagnostic profile digest input. |
| `tests/sol_execbench/fixtures/profile_summary/partial.profile-summary.json` | Partial diagnostics because rocprofv3 failed or profile artifacts are incomplete. | Accept with limitation text and lower confidence. |
| `tests/sol_execbench/fixtures/profile_summary/unavailable.profile-summary.json` | No profile result was available. | Treat as unavailable diagnostic state. |
| `tests/sol_execbench/fixtures/profile_summary/stale.profile-summary.json` | Valid schema with old trace/run identity. | Reject as stale for the current run. |
| `tests/sol_execbench/fixtures/profile_summary/malformed.profile-summary.json` | Invalid JSON payload. | Treat as invalid diagnostic state. |
| `tests/sol_execbench/fixtures/profile_summary/contradictory-authority.profile-summary.json` | Schema-shaped payload with forbidden score authority enum. | Reject before prompt assembly. |
| `tests/sol_execbench/fixtures/profile_summary/missing.profile-summary.case.json` | Metadata fixture for absent sidecar path behavior. | Treat missing sidecar as unavailable. |

These fixtures use synthetic checksums and compact file names. They are not
benchmark evidence and do not represent real profiler output.

## Consumer Rules

1. Parse with strict schema validation.
2. Reject malformed JSON and contradictory authority payloads before prompt
   assembly.
3. Validate freshness against the current trace path and any available run
   identity.
4. Downgrade stale, unknown, missing, unavailable, and partial states instead of
   promoting benchmark claims.
5. Map status, metrics, bottleneck hints, limitations, and citations into a
   closed HIP profile digest taxonomy. Unknown values must be downgraded, not
   promoted.
6. Keep canonical Trace JSONL as the only source for correctness, timing,
   scoring, and status.
