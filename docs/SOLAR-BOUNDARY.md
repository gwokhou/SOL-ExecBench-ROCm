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
  Graph Extractor -> strict extended-einsum converter -> SOL Analyzer
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
public `solar.api` types (`AnalysisResult`, `AnalysisFailure`, `ArtifactRef`,
`SolBound`) to verify the outcome-mapping logic.

Formal conversion is offline and fail-closed. Graph extraction is a declared
route choice: `nvlabs` uses the reviewed Torchview extractor and `mainline`
uses `make_fx`. Both emit the typed operator-artifact contract with exact source
argument indices, tensor metadata, effects, and outputs; the requested IR
representation remains a separate choice. Conversion validates the recorded
extraction provenance and rejects unsupported route/backend pairings before
replay or analysis. Unsupported tracing, conversion, execution, or resource
accounting stops publication.

The maintained NVLABS-derived graph and IR code lives in the first-party
`solar.nvlabs` namespace. It is a deeply adapted implementation, not an
untouched third-party snapshot; `solar._vendor` is reserved for dependencies
that remain vendored.

`AnalysisRequest` composes the same `ConversionRequest` used by readiness
auditing. That conversion request owns one `VerificationPolicy`, so the route,
IR, replay device, seeds, input patterns, and numerical tolerances cannot drift
between the readiness and formal-analysis entry points.

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
