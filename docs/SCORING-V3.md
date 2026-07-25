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

## SOLAR bound policy

The paper (§4.2) treats Orojenesis as an optional tighter-bound path that
lives inside the SOL Analyzer; the Eq. 1 roofline bound alone is the SOL
Analyzer's default formal output. This port follows the paper by default:
`AnalysisRequest.require_orojenesis` defaults to `False`, so `solar.api.analyze`
accepts the Eq. 1 roofline (`bound_kind == "diagnostic"`,
`T_SOL = max(compute, fused_bytes / bandwidth)`) as the bound. Setting
`require_orojenesis=True` restores the port's stricter release-evidence policy,
which requires the capacity-constrained / Orojenesis tile-aware bound
(`bound_kind == "capacity_constrained_tile_aware_v1"`).

Regardless of this flag, formal publication is *additionally* gated by verified
architecture audit evidence (`ArchitectureProfile.require_verified_audit_evidence`),
an independent guard the flag does not bypass. The packaged RX_9060_XT profile
references a content-addressed locked-clock v3 audit. Loading the profile
requires exact coverage of all non-exempt precision and resource calibration
targets, then cross-checks FP32/FP16 VALU and FP16/BF16/FP8/INT8 WMMA claims
against the machine-readable gfx1200 ISA, emitted HIP code objects, and runtime
probes. It also verifies the two-phase tuning/held-out protocol, frozen
configuration, raw samples, telemetry, schema, payload checksum, GPU identity,
clock state, and nominal ceilings.

Diagnostic workload scoring
(`sol_execbench.core.scoring.diagnostic_workload_score`) wraps the paper formula
into an aggregate-able workload score from caller-supplied `T_k`, `T_b`, and
`T_SOL`. It is non-official: the official scorer stays unwired because the
paper's `T_b` baseline and release authority are not published. Passing the
architecture audit gate does not supply those missing evidence classes. See
[SOLAR boundary](SOLAR-BOUNDARY.md) for the cross-package seam.
