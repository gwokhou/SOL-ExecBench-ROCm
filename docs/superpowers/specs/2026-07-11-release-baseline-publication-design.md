# Release Baseline Publication Design

## Purpose

Close report items 7.3 and 7.7 from
`docs/sol_score_gap_and_amd_reuse_report.md`: generate a release-scoped,
provenance-complete `sol_execbench.scoring_baseline.v1` from optimized baseline
measurements, independently rerun it, and publish a versioned AMD score evidence
bundle.  The release baseline covers the complete selected suite.  It does not
silently upgrade incomplete evidence into official score authority.

## Scope and claim boundary

The implementation publishes every workload in the selected suite exactly once
in the release manifest, including failed and evidence-ineligible workloads.
Each workload has one of these classifications:

- `official`: all release baseline, independent-rerun, bound, hardware-model,
  environment, and aggregation requirements are satisfied;
- `derived`: a positive measured baseline is available but a bound, hardware
  model, or other authority requirement is intentionally non-official;
- `blocked`: a required baseline measurement or provenance reference is absent,
  inconsistent, invalid, or stale.

Only `official` rows may feed an official score.  `derived` and `blocked` rows
remain in the published denominator and release summary.  A bundle with either
classification cannot claim that the full suite is official.  Existing
`scoring_baseline.v1` consumers remain compatible and continue to consume its
compact positive-latency entries.

## Chosen architecture

Use one versioned `release_baseline_bundle.v1` as the release unit, containing
and checksumming a separately written `scoring_baseline.v1`.  Keep numerical
score input separate from release audit data: the compact baseline artifact is
the only input to existing scoring code, while the bundle records provenance,
classification, rerun evidence, and publication eligibility.

The bundle contains:

- release ID, schema version, generation time, suite manifest reference and
  checksum;
- baseline artifact reference and checksum;
- fixed optimized solution identity and content hash;
- immutable execution provenance: GPU/hardware identifier, ROCm version,
  clocks, compiler/build options, timing policy, trace reference and trace
  checksum;
- one workload record for every suite workload, keyed by definition and UUID;
- bound/hardware-model references and checksums, authority blockers, and the
  `official`/`derived`/`blocked` classification;
- independent rerun references, values, comparison results, and allowable
  latency tolerance;
- summary counts whose total equals the selected-suite denominator.

The baseline publisher consumes an optimized-baseline trace plus a canonical
suite/workload manifest.  It rejects ambiguous trace records, duplicate
definition/UUID pairs, non-positive or non-finite latency, and mismatched
workload identity as baseline candidates.  Those cases still receive a
`blocked` manifest record rather than disappearing.  Valid records emit compact
`scoring_baseline.v1` entries and carry the complete provenance in the bundle.

## Independent-rerun verification

The verifier consumes the published bundle and a trace collected in a different
run directory.  It compares each workload's definition, UUID, solution hash,
environment fingerprint, clock policy, compiler/build identity, timing policy,
bound/model checksums, and measured latency.  Latency comparison uses a
release-configured relative tolerance recorded in the bundle.  A missing rerun
record, mismatched immutable field, or out-of-tolerance latency marks that
workload `blocked`; a non-authoritative bound/model retains `derived` rather
than becoming official.

The verifier must not mutate the signed/generated baseline values.  It writes a
new verification report referenced by the final bundle so reviewers can
distinguish original measurement from confirmation evidence.

## Publication flow

1. Select and checksum the suite manifest and fixed optimized solution.
2. Run the baseline solution under fixed release environment and timing policy.
3. Generate compact `scoring_baseline.v1` plus preliminary
   `release_baseline_bundle.v1` records for every selected workload.
4. Produce/collect bound and hardware-model eligibility evidence, then classify
   each preliminary record.
5. Run the same candidate/baseline context independently in a new output
   directory and verify immutable evidence and latency tolerance.
6. Finalize classifications and summaries, write deterministic checksums, and
   hand the final bundle to the existing prerelease artifact bundle/readiness
   gates.
7. Publish the versioned bundle and release notes.  The notes state official,
   derived, and blocked counts and prohibit a full-suite-official claim unless
   the latter two counts are zero.

## Failure handling and invariants

- The suite manifest is the authoritative denominator.  Every selected workload
  has exactly one release record and exactly one final classification.
- No missing, non-positive, NaN, or infinite latency enters
  `scoring_baseline.v1`.
- All published referenced files have SHA-256 checksums; a mismatch blocks
  readiness.
- Immutable environment or solution identity drift blocks the affected workload
  even if latency is within tolerance.
- No fallback reference latency, derived score, or measured-baseline registry
  value may be promoted to an official baseline without required release
  provenance and rerun verification.
- Existing prerelease artifacts remain engineering-release evidence; this work
  only establishes the explicit scope recorded in the release bundle.

## Testing strategy

Unit tests cover parsing and deterministic serialization, complete-suite
coverage, duplicate/missing trace behavior, status classification, checksum
drift, non-finite latency rejection, environment/solution/bound mismatch, and
relative latency tolerance boundaries.  CLI tests cover baseline build and
rerun-verify commands.  Integration tests build a small synthetic suite,
generate and verify a bundle in separate directories, and prove that the
readiness gate accepts a complete internally consistent release while retaining
derived/blocked counts and rejecting an over-broad authority claim.

## Non-goals

This work does not calibrate `gfx1200`, change AMD SOL/SOLAR mathematics, make
all operator bounds exact, establish NVIDIA leaderboard equivalence, or run a
full hardware release in CI.  It makes the existing evidence boundary
publishable, explicit, and independently verifiable.
