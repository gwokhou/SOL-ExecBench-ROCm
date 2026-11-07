# Profile Summary Sidecar

`sol_execbench.profile_summary.v5` is an optional diagnostic sidecar written
next to canonical Trace JSONL using the profile-summary JSON suffix. It normalizes
bounded `rocprofv3` profile metadata for downstream adapters while preserving
Trace JSONL as the only authority for correctness, timing, scoring, and
evaluation status.

The evaluator contract advertises this artifact through the optional
`profile_summary.sidecar` capability key. The concrete artifact schema remains
`sol_execbench.profile_summary.v5`.

The summary sidecar does not replace the raw profile JSON sidecar. The existing
profile metadata sidecar remains the raw diagnostic metadata record and is cited
from the normalized profile summary when present.

When a `rocprofv3` profile result is supplied, the summary carries bounded
artifact registration metadata from the `profile` variant of
`sol_execbench.rocprofv3_session.v1`:
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

`summary.hip_events[]` contains bounded GPU barrier observations from registered
ROCProfiler JSON or rocpd 3.0.4 databases. Each record retains capture, process,
agent, queue, optional stream, event handle, source queue, operation, timestamps,
and an optional observed record generation. The readers limit JSON to 16 MB,
databases to 128 MB, and events to 10,000 per artifact; SQLite is read-only and
has a query instruction budget. Other database contents, Perfetto/PFTrace and
OTF2 remain citation-only.

Generations are ordinals within the observed capture, not proven event lifetimes.
Wait generations remain unknown because GPU completion timestamps cannot identify
the host record call consumed by a wait. Reused handles, duplicate records and
unresolved waits produce parse warnings. JSON and rocpd copies remain separately
scoped captures; consumers must not add their event counts together. A barrier's
duration is not a measurement of avoidable synchronization overhead.

`summary.hip_event_causality[]` adds conditional event-to-dispatch links from
successful collections with ROCProfiler JSON host arguments and return values.
The default runtime-enabled rocpd diagnostic command also writes JSON for this
purpose. Each link uses host correlation IDs and submission order to connect the
last producer dispatch before RECORD to the first consumer dispatch after WAIT.
It retains the source SHA-256, capture/process identity, event handle, device,
queues, streams, and all four host call IDs. It does not reinterpret the GPU-only
`hip_events[].generation` field or join JSON to rocpd rows.

The supported subset is a single device and host submission thread with explicit
nondefault streams, successful ordinary record/wait/launch calls, and completed
rerecords. Missing identities, duplicate IDs, failed calls, ambiguous order,
unsupported graphs/IPC/PTDS or incomplete observations cannot supply a link.
Unresolved waits and rejection codes remain visible. Even a capture with all
observed waits linked has `status: partial`: dropped records cannot be ruled out,
and dispatch IDs have not been bound to the complete workload/candidate scope.
Consumers must preserve these limitations and must not infer that unlinked
work is independent. An explicitly successful `hipEventSynchronize` for the same
event can establish completion before rerecording; an event observation timestamp
alone is not an exact completion fence. See the [acceptance criteria and evidence boundary](../internal/hip-event-causality-acceptance.md).

The performance model's nested schedule evidence
retains only verified precedence: same-lane order, plus cross-lane
`event_wait` edges when an independently qualified scope schedule supplied
verified event links. Timestamp order across lanes still does not establish a
dependency. Multi-lane predictions without that qualification return
`cross_lane_event_dependencies_unverified`; receiving observations or
conditional diagnostic links does not remove that restriction.

Schedule identity and lane checks include every observed dispatch, even a helper
whose counters are excluded from the hardware estimate. Filtering such a helper
must not turn a multi-stream capture into an apparently serial workload. Mixed
workload or candidate identities cannot be combined into one prediction. Same-lane
overlap, tied starts or incomplete ordering timestamps do not produce precedence
edges. A single-lane estimate can still sum calibrated dispatch costs without
using timestamp distances as durations.

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
`sol_execbench.profile_summary.v5` sidecar; SOL does not surface them as
`agent_feedback.items[]`. HIP adapters that want one prompt-facing hint list
should merge profile-summary hints with agent-feedback items only after each
source sidecar passes strict freshness and authority checks. The separate
`hip_event_observations` agent-feedback item is informational and cites its
captures. It does not classify a bottleneck, supersede static Decision hints,
or admit a code-changing action.

HIP runtime prompts should never include raw trace rows, raw profiler dumps,
full kernel source, prompt text, or absolute temporary paths from SOL profile
summaries. They should include only normalized status, bounded metrics,
limitations, and compact citations after freshness and authority checks pass.
Profiler artifact citations use compact file names and compute SHA256 by
default. Hashing large `.rocpd` or database artifacts can be expensive, but the
current contract has no size-based hashing exemption. HIP consumers must not
infer one.

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


## Controlled executor scope qualification

The separate `sol_execbench.hip_trace_scope.v1` artifact checks a controller-owned
run/workload/candidate identity against an independent
`sol_execbench.hip_trace_journal.v1` application journal. It requires a bijection
for the audited launch/event call domains and binds every dispatch, including
otherwise independent work. Missing or extra rows, identity conflicts, failed
calls, incomplete journals and missing completion fences reject qualification.

This is conditional on a trusted, audited native executor; an arbitrary candidate
cannot grant itself this authority by writing a journal. The ordinary summary
continues to report partial causal observations. A qualified controlled scope
also has `schedule_complete=false` and `performance_authority=false`; it does not
cover every dependency domain or enable cross-stream prediction. See the
[scope acceptance report](../internal/hip-trace-scope-acceptance.md) for the exact
conditions, measured coverage, and remaining boundary.

The separate `sol_execbench.hip_torch_scope.v2` qualification extends this to an
audited eager PyTorch executor. A `sol_execbench.hip_sdk_ledger.v2` collector
records SDK enqueue callbacks and LOSSLESS buffered completions, requires zero
drops and explicit stop/flush, and checks the entire exported dispatch set.
Internal memory helper kernels are retained, including reported stream zero.
Workload membership comes from executor boundaries and submission order.
Engine copies are a qualified dependency domain: the collector records
`memory_copy` completions and the raw stream argument of the frozen async
memcpy core set; qualification requires ledger/export/API three-way agreement
and a co-observed stream-to-queue lane binding. Within this restricted domain
`schedule_complete` may be true — never when a legacy default stream mixes
with other lanes, when a copy lane is unresolved, or when an event wait stays
unresolved. Graph capture, IPC, cross-process work and background-thread
submissions remain rejected, and `performance_authority=false` always.

This opt-in path covers a single process/GPU/submission thread and one record
generation per event in scope. No-op waits without a GPU barrier remain
unqualified. Callback and buffer channels share the SDK/runtime; common omissions
are not excluded. The ordinary summary remains partial, and neither the separate
artifact nor its proximity on disk authorizes timing, cross-stream prediction,
or synchronization removal. See [PyTorch scope acceptance](../internal/hip-torch-scope-acceptance.md)
and the [schedule-causality acceptance](../internal/solar-l2-schedule-causality-acceptance-2026-11-06.md).


The production `evaluate --profile rocprofv3-scope --trace-output TRACE.jsonl`
mode replays each workload independently after the ordinary evaluation, binding
identity from workload content, candidate source, executor, and physical GPU.
It requires a single `ROCR_VISIBLE_DEVICES` UUID and a verified STABLE_PEAK
window, and writes `sol_execbench.hip_evaluate_intent.v1`,
`sol_execbench.hip_evaluate_receipt.v1`, and
`sol_execbench.hip_evaluate_scope.v2` artifacts (reusing the
`sol_execbench.hip_sdk_ledger.v2` collector) under `TRACE.jsonl.hip-scope/<run>/`
without reusing canonical timing samples. When any workload lacks evidence or is
unsupported, its qualification result is `rejected` and the CLI fails, while the
completed canonical trace is preserved. `all_passed` and `scope_qualified`
report the two outcomes separately. The ordinary profile summary keeps its own
authority and never scans that archive to promote qualification. See the
[evaluate scope acceptance report](../internal/evaluate-hip-scope-acceptance.md).
