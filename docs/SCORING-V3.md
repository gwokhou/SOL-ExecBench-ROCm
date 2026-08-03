# Scoring contract

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

The checked-in schema v7 corpus defines 43 scored problems and 163 scored
workloads. The verifier accepts only a publisher release bundle that binds the
canonical baseline run, candidate run, per-workload SOLAR manifests, public
corpus, and architecture identities. It does not accept caller-supplied runtime
JSON.

`score status` reports four separate states instead of a single overloaded
availability flag: verifier availability, corpus-policy authorization, formal
producer readiness, and whether a repository release bundle is published. The
verifier and formal producer are available, but the v2 baseline evidence has
not been published. The schema v7 manifest therefore keeps policy authorization
fail-closed with reason `baseline_v2_release_evidence_pending`.

## SOLAR bound policy

The paper (§4.2) treats Orojenesis as an optional tighter-bound path that
lives inside the SOL Analyzer; the Eq. 1 roofline bound alone is the SOL
Analyzer's default formal output. This port follows the paper by default:
`AnalysisRequest.require_orojenesis` defaults to `False`, so `solar.api.analyze`
accepts the Eq. 1 roofline (`bound_kind == "roofline_eq1_v1"`,
`T_SOL = max(compute, fused_bytes / bandwidth)`) as the bound. Setting
`require_orojenesis=True` restores the port's stricter release-evidence policy,
which requires the capacity-constrained / Orojenesis tile-aware bound
(`bound_kind == "capacity_constrained_tile_aware_v1"`).

The outer `sol-execbench solar analyze` command is the formal publication
surface and always sets `require_orojenesis=True`. Its worker IPC, bridge, and
CLI independently reject an analyzed response unless it carries the exact
capacity-constrained bound, a positive finite lower bound, the complete
content-addressed artifact set, and `publication_eligible=true`. The paper-level
Eq. 1 result remains available through the benchmark-agnostic Python API and
has `sol_score_eligible=true`, but cannot cross this port's stricter formal
bridge.

`solar_request_manifest` schema 2 records `require_orojenesis`,
`sol_score_eligible`, and `publication_eligible`. The first eligibility flag
tracks the paper's SOL Score semantics; the second tracks this port's stricter
release policy. Formal Orojenesis acceptance requires the
repository-allowlisted mapper binary plus the pinned provenance manifest,
source archive/tree, builder image, compiler-wrapper digest, and compiler
identity. A git checkout without the provenance manifest is not accepted.
There is no git-checkout fallback. The current allowlist contains the digest
produced identically by two no-cache builds from the pinned source and build
environment. Any other mapper still fails closed.

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
`T_SOL`. It is non-official. The official scorer is separately wired to accept
only the exact corpus plus content-addressed baseline, candidate, and SOLAR
statements from a publisher bundle. Passing the architecture audit gate does
not supply those release evidence classes. See
[SOLAR boundary](SOLAR-BOUNDARY.md) for the cross-package seam.
