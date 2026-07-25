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

`solar` may depend on PyTorch, torchview, graph conversion code, architecture
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

Formal conversion is offline and fail-closed. The converter reads generated
handlers only from `src/solar/handlers/`. The learning command writes candidates
elsewhere and cannot activate them automatically. Formal lookup accepts only
records with passed verification, `formal_review: approved`, matching metadata
and source SHA-256 values, and a safe package-relative source path.
Graph extraction also fails closed before conversion when any tensor dispatch
loses torchview `RecorderTensor` lineage. This covers both fully empty traces
and partial graphs in which only some operations remain visible; SOLAR never
publishes a lower bound from the surviving subset.

The ROCm formal-publication profile uses a pinned Orojenesis mapper when the
stricter capacity-constrained bound is requested
(`AnalysisRequest.require_orojenesis=True`). The default bound policy follows
the paper and accepts the Eq. 1 roofline. Either way this is a release-evidence
policy of this port, not an expansion of SOLAR into benchmark evaluation or a
claim of universal paper parity.

The benchmark-agnostic Python API retains that diagnostic default. The outer
`sol-execbench solar analyze` bridge is stricter: it always requires
Orojenesis and rejects any worker response that is not an explicitly
publication-eligible `capacity_constrained_tile_aware_v1` result with the
complete artifact set. The request manifest records the policy and result.
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
allows SOLAR analysis to proceed; it does not publish the independent baseline,
rerun, execution-attestation, or release-authority evidence required for an
official SOL ExecBench score. Generic candidate evaluation remains diagnostic.
