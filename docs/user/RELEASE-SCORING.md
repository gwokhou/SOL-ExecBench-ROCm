# Release and Official Score Workflow

This guide connects the advanced evidence commands into one auditable workflow.
It is for release owners and integrators, not a replacement for a normal
`evaluate` run. Canonical Trace JSONL remains the evaluation artifact; profiles,
calibrations, fusion records, and sidecars have only the authority stated in
their own schemas.

## What This Workflow Produces

The final command can write `official_score_evidence.v1`. A non-null confirmed
score requires all of the following:

1. An AMD-native suite report produced from canonical traces and its required
   bound/model evidence.
2. A frozen `scoring_baseline.v1`, matching release-baseline bundle, and a
   passing independent rerun verification.
3. A canonical suite manifest that fixes the denominator and an explicit
   `fixed_suite_denominator_zero_for_blocked` aggregation policy.

`baseline export` and `baseline compare` are useful local tools, but are not a
release scoring baseline. A calibration, profiler result, static sidecar, or
partial artifact set also cannot create score authority by itself. See
[Confirmed Evidence](confirmed_evidence.md) for the consumer contract and
[Claims](CLAIMS.md) for forbidden upgrades.

## 1. Collect Hardware and Fusion Inputs

On the target AMD host, collect calibration evidence. This command writes a
rejected diagnostic artifact rather than inventing values when prerequisites
are unavailable:

```bash
uv run sol-execbench hardware model calibrate \
  --device 0 --architecture gfx1200 --require-clock-lock \
  --output out/calibration.json
```

When the score suite defines exact hardware-profile requirements, pass its
requirements file to calibration. Convert only validated calibration evidence
into an external v3 hardware model:

```bash
uv run sol-execbench hardware model build \
  --calibration out/calibration.json \
  --requirements out/hardware-profile-requirements.json \
  --verification-calibration out/verification-calibration.json \
  --output out/hardware-model.json
```

The build command may require additional shape-aware evidence, coverage, and
plan inputs for an authority envelope. Use `--help` for that release's exact
requirements; do not omit them merely to obtain a file.

Collect and verify shape-exact fusion capacity against the same suite and
benchmark root:

```bash
uv run sol-execbench hardware fusion collect \
  --device 0 --architecture gfx1200 \
  --suite-manifest release/suite.json --benchmark-root data/benchmark \
  --require-clock-lock --output out/fusion-validation.json
uv run sol-execbench hardware fusion verify \
  --evidence out/fusion-validation.json --suite-manifest release/suite.json
```

These hardware and fusion artifacts are inputs to validated AMD SOL bound
generation. They do not by themselves establish a score or wider hardware
validation claim. See [AMD SOL Bound Evidence](amd_sol.md) for the accepted
bound schema.

## 2. Produce and Freeze Release Inputs

Run the release's canonical workload set and retain canonical trace JSONL,
timing evidence, solution identity, environment fingerprint, clock policy,
compiler build ID, and timing policy. Generate the AMD-native report and v5
bound artifacts through the dataset workflow; the bounded local example is in
the [Cookbook](COOKBOOK.md#recipe-generate-amd-native-derived-evidence).

The release owner then freezes the suite denominator and builds a baseline. The
following is a parameterized skeleton; replace every angle-bracket value with
the exact, recorded release value:

```bash
uv run sol-execbench baseline release build \
  --suite-manifest release/suite.json --trace out/baseline.trace.jsonl \
  --release <release-id> \
  --baseline-output out/scoring-baseline.json \
  --bundle-output out/release-baseline-bundle.json \
  --authority-json out/release-authority.json \
  --solution <baseline-solution-id> --solution-sha256 <64-hex-sha256> \
  --environment-fingerprint <environment-fingerprint> \
  --clock-policy <clock-policy> --timing-policy <timing-policy> \
  --compiler-build-id <compiler-build-id> \
  --scope <bounded-release-scope> --latency-tolerance-rel <positive-fraction>
```

Do not label the bundle official until an independent rerun with matching
provenance has passed:

```bash
uv run sol-execbench baseline release verify \
  --bundle out/release-baseline-bundle.json \
  --rerun-trace out/baseline-rerun.trace.jsonl \
  --output out/release-baseline-verification.json \
  --solution-sha256 <64-hex-sha256> \
  --environment-fingerprint <environment-fingerprint> \
  --clock-policy <clock-policy> --timing-policy <timing-policy> \
  --compiler-build-id <compiler-build-id> \
  --suite-manifest-sha256 <64-hex-sha256>
```

The `scope` is mandatory and must remain attached to downstream claims. It
prevents an authority slice from being described as a full benchmark suite.

## 3. Emit Official Score Evidence

Use the AMD-native suite report together with the baseline, bundle, passing
verification, and the same suite manifest:

```bash
uv run sol-execbench score official \
  --amd-native-score out/amd-native-score.json \
  --scoring-baseline out/scoring-baseline.json \
  --release-baseline-bundle out/release-baseline-bundle.json \
  --release-baseline-verification out/release-baseline-verification.json \
  --suite-manifest release/suite.json \
  --aggregation-policy fixed_suite_denominator_zero_for_blocked \
  --output out/official-score-evidence.json
```

For a candidate-specific confirmed score, provide all six candidate evidence
options together: `--candidate-solution`, `--candidate-trace`,
`--candidate-timing-evidence`, `--candidate-environment-fingerprint`,
`--candidate-clock-policy`, and `--candidate-timing-policy`. Supplying only a
subset is rejected. `--measured-registry` adds coverage diagnostics; it is not
the scoring baseline.

## Interpret and Publish Safely

Read `score_authority`, `blocker_reason_codes`, the workload counts, and the
recorded scope from the emitted file. A blocked, unscored, partial, or
diagnostic result must remain so in reports. It does not imply full 235-problem
paper validation, upstream SOLAR parity, native-host validation inferred from a
container, or leaderboard readiness.

Before publishing, follow [Public Prerelease](public_prerelease.md), preserve
the evidence references and checksums, and state all unresolved boundaries from
[Claims](CLAIMS.md). For the exact machine-readable surface of a checked-out
version, run `uv run sol-execbench contract cli` and the relevant `--help`
command; those commands are authoritative when a future CLI release changes.
