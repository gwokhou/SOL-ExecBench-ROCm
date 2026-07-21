# Scoring contract v3

SOLAR emits a lower-bound runtime and no candidate-facing metric. The outer
benchmark applies the paper score:

```text
S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))
```

For correct candidates, all runtimes must be positive and finite,
`T_b > T_SOL`, and `T_k >= T_SOL`. Violating either ordering is an audit signal;
the implementation raises an error and does not clip, substitute, or silently
degrade the score. Incorrect candidates receive zero.

The public corpus is an AKA-derived seed set of scored problems. Workload
scores belonging to the same problem are averaged first. Those per-problem
means are then averaged with equal problem weight. Problems flagged as
compatibility sentinels never enter either denominator.

The checked-in v3 corpus explicitly marks official scoring unavailable because
no release baseline, independent rerun, trusted candidate execution
attestation, or pinned SOLAR manifest set has been published for these
problems. The command fails closed instead of treating caller-authored JSON as
authority. A future release must pin and verify all four evidence classes plus
the public corpus and architecture identities before enabling official output.

## SOLAR formal bound policy (stricter than the paper)

The paper (§4.2) treats Orojenesis as an optional tighter-bound path that
lives inside the SOL Analyzer; the Eq. 1 roofline bound alone is the SOL
Analyzer's default formal output. This port's formal publication path is
deliberately stricter: `solar.api.analyze` requires Orojenesis
(`require_orojenesis=True` in `_run_analysis`) and rejects any non-Orojenesis
result as non-formal (`bound_kind` must be `capacity_constrained_tile_aware_v1`).
The Eq. 1-only roofline seconds are still computed and exposed as
`lower_bound_components` diagnostic data, so the formula is implemented
faithfully — but the port never publishes an Eq. 1-only `T_SOL` as a formal
bound. This is a release-evidence policy of the ROCm port, not a defect in the
bound derivation and not an expansion of SOLAR into scoring. See
[SOLAR boundary](SOLAR-BOUNDARY.md) for the cross-package seam that enforces it.
