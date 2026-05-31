# Phase 91 Context: Claim Upgrade Rules And Authority Gates

## Objective

Define machine-readable claim-upgrade prerequisites and evaluate whether the
current evidence set can support stronger validation/authority claims.

## Inputs

- `consistency_report.v1` from Phase 89.
- `evaluation_stability.v1` from Phase 90.
- Existing closure, denominator, Matrix, AMD score, AMD SOL/SOLAR, AMD bound
  sanity, and optional hardware-validation refs.

## Boundaries

- The evaluator must not mutate any source report or upgrade authority fields
  inside v1.19/v1.20 artifacts.
- The output is a sidecar report explaining eligible levels, blocked levels,
  unmet prerequisites, and next evidence hints.
- Full paper parity, hosted leaderboard readiness, native-host Matrix
  authority, and new hardware validation remain deferred unless explicit
  external evidence is supplied.

