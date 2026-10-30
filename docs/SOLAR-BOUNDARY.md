# SOLAR responsibility boundary

The boundary follows SOL-ExecBench §4.2 and is enforced by import tests.

```text
public problems + workload + reference
                 |
                 v
sol_execbench.core.solar_bridge  (the only outer package allowed to import solar)
                 |
                 v
solar.api
  architecture -> graph extraction -> strict extended-einsum conversion
  -> conversion verification -> formal analysis
                 |
                 v
hash-bound graphs, conversion proof, formal lower bound
```

`solar` may depend on PyTorch graph capture, semantic graph code, architecture
profiles, and Orojenesis. It must not import `sol_execbench` or model benchmark
definitions, workloads, solutions, candidate timing, baselines, or scores.

`sol_execbench` owns pinned upstream-dataset acquisition and audit, AMD compatibility, seeded
input generation, solution compilation and execution, correctness, candidate
timing, scoring baselines, SOL Score, aggregation, and the user CLI. Production
imports of `solar` are confined to `core/solar_bridge/`, whose worker is isolated
with a timeout, process-group cleanup, and file-backed redacted logs. The
boundary test also scans `tests/sol_execbench/` so test code reaches `solar`
only through the bridge, with one documented exception: the bridge's own
contract tests under `tests/sol_execbench/core/solar_bridge/` may reference
SOLAR public contracts and workflows to verify the adapter, outcome mapping
and isolated worker behavior.

Formal conversion is offline and fail-closed. Graph extraction is a declared
`ExtractionKind`: `torchview_v1` uses the reviewed Torchview extractor and
`make_fx_reference_v1` uses `make_fx`. Both emit operator-graph schema v2 with
exact source argument indices, tensor metadata, effects, and outputs. The
requested `IRKind` remains a separate choice. Conversion validates recorded
extraction provenance and rejects unsupported extractor/IR pairings before
replay or analysis.

The maintained extended-einsum implementation lives under
`solar.ir.extended_einsum`. Schema v6 uses only `input`, executable `einsum`,
and canonical native `operation` semantics. Ordered operands, named public-API
attributes, bounded dynamic dimensions, aliases, and mutations are explicit in
the artifact. Its independent executor uses public PyTorch APIs as the default
runtime backend; Extended artifacts do not contain ATen targets or overloads
and never dispatch through the ATen executor. It owns one canonical reviewed
native-operation registry;
the retired `solar.nvlabs` and `solar.einsum` duplicate namespaces no longer
exist. `solar._vendor` remains reserved for dependencies retained as vendored
snapshots.

`AnalysisRequest` composes the same `ConversionRequest` used by readiness
auditing. That conversion request owns one `VerificationPolicy`, so the extractor,
IR, replay device, seeds, input patterns, and numerical tolerances cannot drift
between the readiness and formal-analysis entry points.

Current conversion artifacts include `access.json` (access schema v1), binding
graph digests, logical base/view maps, unknown access reasons, and source storage
layouts. The current IR verifier (`solar.verification.ir.v8`) checks used-input
independence and trace/replay layout agreement before cloning inputs, and binds
the access artifact as an attestation subject. Request manifest schema v7
requires this artifact in publication and comparison inventories.
The verification execution record binds Python implementation/version, Torch
version and the semantic interpreter source digest. Missing or changed runtime
identity requires fresh verification; an old passing flag alone cannot be reused.

Analysis schema v6 requires `metadata.external_reads`: `exact`, `lower_bound`
with union-budget reasons, or `unavailable` with absent numeric totals. Proven
base unions replace external input reads using each base's dtype width. Proven
views have no independent memory traffic; output materialization, initialization,
effects and capacity traffic remain separate obligations. This proof describes
complete operand reads in the fixed operator graph, before algebraic rewriting
or demand propagation. Dynamic indexing, effects, copies, output aliases and
unreviewed read semantics retain diagnostic counts without an eligible bound.
Convolution footprints are not yet proven by the static-view union model;
conversion success does not establish this separate accounting obligation.

The same current analysis schema also requires `metadata.views`, containing
`unfused`, `ideal_fused` and `cache_aware`. Each records availability, reason,
assumptions, read-proof quality, time and traffic. Unfused time sums the per-layer
maximum of compute and memory; ideal fused time takes the global maximum after
summing modes sharing a resource. Cache-aware uses the existing audited traffic
composition and is unavailable when its supported capacity domain is incomplete.
Unavailable views have no numeric time or traffic; partial evidence remains in
`metadata.orojenesis`. These views do not replace the existing formal bound field
or confer publication eligibility.

Internal analysis now separates an owned intrinsic-count snapshot from resource
attribution, capacity evidence and pricing. Reprojection does not reload or
recount the graph; profile-dependent attribution and capacity auditing run again.
The snapshot retains the digest of the exact loaded bytes. Graph selection is
explicit: a neighboring `*_reordered.yaml` is never substituted. The public `solar.reuse.ReuseSession` now connects this seam to a pinned,
complete source bundle. Its request types live in `solar.reuse_contracts`:
`VerifiedProgramRequest`, `ProjectionRequest` and the versioned `ReuseJob`.
Loading checks the entire inventory, current schemas, source/path/policy/runtime
identity and independent access maps before counting. It does not execute the
reference, retrace or load pickle. Each session computes intrinsic work once;
rate, resource-mode and capacity scenarios recompute attribution, capacity
evidence and pricing. Installed analysis-rule fingerprints and all intrinsic,
mapper and output budgets enter derived identities. Rule drift requires a new
session; the implementation conservatively invalidates all intrinsic work when
any analysis implementation changes.

`sol-execbench solar reprice job.json --timeout 900` runs a `ReuseJob` through an
isolated CPU-capable worker with bounded output and process-group cleanup. It
writes a new directory containing `solar-analysis.yaml`, all emitted capacity
evidence, and the current reuse schema v1 `reuse.json`. Results bind their parent
manifest and complete derived inventory; partial output, extra files or digest
drift are rejected by the pinned result loader. Repricing is an analysis command;
it does not execute a candidate or change release acceptance. All three views
and unavailable reasons are displayed by both `solar analyze` and `solar reprice`.
Path-comparison schema v2 also reports the three views separately from the
existing authoritative formal-bound comparison, without selecting a better path.

A changed profile parameter or qualification input (including a rate-only change),
or an explicitly hypothetical request, discards inherited audit fields and has no
score/publication eligibility. Changes only to `source`, `resource_limit_sources`
or a memory level's `source` description preserve the source scenario. Complete
profile digests still record those edits in the manifest and derived result;
audit references, exemptions, support claims and revisions remain qualification
inputs. The source profile is read from the hash-checked analysis bundle.
Even an identical profile must pass a freshly requested resource-audit check to
retain publication eligibility; it cannot gain eligibility from its parent.
The ordinary reuse default leaves audit qualification false and allows model
exploration. The original formal `solar analyze` gate is unchanged. Synthetic
capacity fixtures remain test evidence, not trusted mapper or device receipts.

The public `solar.explore` entry points add fixed-instance necessary resource
queries and bounded profile scans. Exploration schema v1 pins complete derived
results, exact model/view identity, budgets and resume inputs. `solar requirements`
solves only ideal-fused or complete cache-aware constraints at fixed capacity;
unfused inversion is explicitly unsupported. Constraints are necessary conditions,
not device performance guarantees. `solar scan` runs a serial isolated worker,
counts the source once per attempt, and recomputes attribution/capacity/pricing
for every grid point. Every point is hypothetical, including the baseline-rate
point, and has no score or publication eligibility. All planned statuses and
three views remain visible; resumed complete points are revalidated and retain
references to prior immutable directories. Runtime, mapper-thread, rule, profile,
grid or budget drift invalidates reuse. See the [CPU checkpoint](internal/solar-resume-2026-10-26.md)
for supported axes, units, limits and the current acceptance status.

Restricted computation is opt-in through `ProjectionRequest.restricted`.
The original `solar-analysis.yaml` remains dense; a separately bound
`restricted-computation.json` reports structural-zero and backward-demand
counts, traffic, resources, assumptions and availability. Its default IEEE
domain is not applicable to zero elimination. The caller must explicitly assert
`finite_no_overflow_ignore_signed_zero`; sample verification does not prove
that domain. Static triangular masks, reviewed views, pointwise operations and
binary contractions have bounded coordinate rules; unknown operations, copies,
effects and dynamic indices retain full demand. Returned internal views share
their producer's output-storage union; external output aliases remain unproved.
Transformed cache-aware estimates require new applicable capacity evidence.
`solar requirements` and `solar scan` select this model only with
`computation="restricted"`, and scan requests also need the explicit projection
constraints. Selection, constraints and rule/budget identities bind recovery.
Unavailable restricted estimates never fall back to dense. Restricted results
have no score, publication or measured-performance authority. See the
[restricted computation contract](internal/solar-resume-2026-10-26.md).

Contraction-order exploration is exposed by `solar.order.search_contractions`
and `solar search-orders REQUEST.json`. It currently supports ordered, pure,
fixed-shape two-dimensional matrix chains. Every association is extracted on
the same IR path and verified against the explicitly supplied original callable,
using its unchanged seeds, patterns, tolerances and gradient policy. The source
IR is not a numerical oracle. Original baseline and candidate projections use
the same model/profile; they remain hypothetical and carry no score, publication
or measured-performance authority. The selected computation scenario retains
all stage 4 constraints and unavailable states.

The search retains the full Catalan denominator, failed numerical candidates,
unavailable views, candidate/time/work budget stops and stable original-first
ties. Pinned recovery validates complete candidate inventories, original-source
verification, association identity and projections; it cannot mix paths,
profiles, policies or rules. The CLI executes a hash-pinned local reference
module in a bounded CPU worker. See the
[search contract and CPU evidence](internal/solar-resume-2026-10-26.md).

The `--backend` IR identity is one of two fixed paths:
`torchview_extended_einsum` or `make_fx_aten`. The default remains
`torchview_extended_einsum`. Extractor and
IR dialect are not separately selectable, and analysis never retries a failed
workload through the other path. Worker requests, attestations, manifests,
corpus matrices, and release indexes bind the selected path and its canonical
IR artifact name.

The ROCm formal-publication profile uses a pinned Orojenesis mapper when the
stricter capacity-constrained bound is requested
(`AnalysisRequest.require_orojenesis=True`). The default bound policy follows
the paper and accepts the Eq. 1 roofline. Either way this is a release-evidence
policy of this port, not an expansion of SOLAR into benchmark evaluation or a
claim of universal paper parity.

The benchmark-agnostic Python API emits that paper-defined default as
`roofline_eq1_v1`; it is a valid `T_SOL` and is marked `sol_score_eligible`.
The outer `sol-execbench solar analyze` bridge applies a stricter port-specific
release policy: it always requires Orojenesis and rejects any worker response
that is not an explicitly publication-eligible
`capacity_constrained_tile_aware_v1` result with the complete artifact set.
The request manifest records both the paper-level eligibility and the stricter
release decision.
Formal toolchain verification requires a reviewed binary allowlist entry and
the pinned provenance/build identity; there is no git-checkout fallback.

The only intended formal target is the packaged `RX_9060_XT` profile and an
observed ROCm `gfx1200` device. Its referenced locked-clock resource-audit file
is content-addressed and validated before graph extraction. Its sole supported
v3 contract must exactly match every non-exempt precision and resource mode in
the profile, and requires frozen tuning plus independent held-out raw samples.
FP32/FP16 VALU and FP16/BF16/FP8/INT8 WMMA claims each require machine-readable
ISA presence, compiler-emitted code-object instructions, and successful runtime
probes. Packed-BF16 VALU is separately proven absent with an emitted FP32
fallback. Passing this architecture gate
allows SOLAR analysis to proceed; it does not publish the canonical baseline,
candidate execution, or per-workload SOLAR evidence required for an official
SOL ExecBench score. Generic candidate evaluation remains outside SOLAR.
