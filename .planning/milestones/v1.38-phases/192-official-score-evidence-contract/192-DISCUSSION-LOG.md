# Phase 192 Discussion Log

**Date:** 2026-06-21
**Phase:** 192 - Official Score Evidence Contract

## Questions Asked

### 1. Relationship To AMD-native Score

Options:

1. Add a separate `official_score_evidence` schema and reference existing
   AMD-native/SOL evidence as inputs.
2. Extend existing `amd_native_score.v1`.
3. Wrap existing score only at report level without a new core schema.

User selected: **1**

Decision: Phase 192 adds a separate official score evidence schema/report.

### 2. Non-null Official Score Preconditions

Options:

1. Require measured latency, official measured baseline, SOL/SOLAR bound, and
   aggregation policy.
2. Temporarily allow baseline artifact or `reference_latency_ms`, but label
   source.
3. Produce official score whenever existing AMD-native score is non-null.

User selected: **1**

Decision: Non-null official score requires all confirmed score inputs.

### 3. Placeholder/Reference Baseline Handling

Options:

1. Block confirmed/official score and emit blocker reason code.
2. Emit null score and classify as non-confirmed.
3. Allow provisional score separately from official score.

User selected: **1**

Decision: `reference_latency_ms` and placeholder/reference baseline fallback
block official confirmed score claims with stable blocker reason codes.

## Locked Decisions

- Official score evidence is separate from AMD-native provisional score.
- Official score must be non-null only when confirmed inputs are present.
- Reference/placeholder baseline cannot satisfy official score baseline
  authority.
- Existing diagnostic speedup and AMD-native score remain separate evidence
  surfaces.
