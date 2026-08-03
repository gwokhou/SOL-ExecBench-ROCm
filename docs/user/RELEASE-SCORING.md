# Release and Official Score Workflow

Canonical Trace JSONL is the execution artifact. A numeric formula result is
official only when it is derived from the repository-pinned corpus and a
publisher-authored, content-addressed release bundle.

## Pending v2 scoring contract

The schema v7 RX 9060 XT corpus contains 43 scored problems and 163 scored
workloads. The FP8 compatibility sentinel and the provenance-retained
`l2n55_matmul_maxpool_sum_scale` target-incompatible problem are excluded from
the denominator. The latter keeps its original AKA shapes; it is excluded
because its trusted-reference IPC payload exceeds the bounded protocol.

The v2 baseline evidence has not been published yet, so the manifest
deliberately keeps official scoring fail-closed:

```yaml
official_scoring:
  status: unavailable
  baseline_id: rx9060xt-gfx1200-reference-v2
  reason_code: baseline_v2_release_evidence_pending
```

The release policy follows the paper's release-defined baseline `T_b`,
candidate runtime `T_k`, formal bound `T_SOL`, and controlled harness. It does
not require multiple release roles, detached signatures, or a second baseline
run.

Content hashes prove that bundle artifacts have not changed. They do not prove
who published a bundle. Publisher authenticity therefore comes from the
repository or release channel that distributes the bundle.

## Reproduce the formal mapper

Before generating formal manifests, build the pinned mapper twice from the
digest-pinned builder and Ubuntu snapshot:

```bash
scripts/internal/orojenesis/verify_reproducible_build.sh \
  out/orojenesis-reproducible
```

The command uses two clean builds and publishes the first artifact only when
the mapper and provenance are byte-identical. A reviewer must inspect the
printed digest and provenance before adding that digest to the repository-owned
`OROJENESIS_TRUSTED_MAPPER_SHA256` allowlist. A locally self-declared
provenance file is insufficient.

## Build and execute the release

Create the canonical eager-PyTorch baseline:

```bash
uv run sol-execbench baseline release-build out/release \
  --baseline-id rx9060xt-gfx1200-reference-v2 \
  --source-revision SOURCE_GIT_SHA
```

The builder derives every baseline `solution.json` from the exact
corpus-pinned Definition reference. The verifier reconstructs that canonical
solution and rejects any baseline implementation difference.

Run the baseline inside the hardened container on the pinned GPU:

```bash
./scripts/run_docker.sh -- sol-execbench baseline release-run \
  /outputs/release/baseline/plan.json
```

The wrapper records the immutable Docker image ID. The runner also records the
clean source revision and exact runtime identity.

Candidate input contains one `solution.json` below every scored problem path:

```bash
uv run sol-execbench baseline candidate-build out/release CANDIDATE_ROOT \
  --candidate-id CANDIDATE_ID \
  --source-revision SOURCE_GIT_SHA
./scripts/run_docker.sh -- sol-execbench baseline release-run \
  /outputs/release/candidate/plan.json
```

Baseline and candidate must use the same source revision and validated runtime
environment, including the immutable container image. A candidate failure is
retained in its complete trace denominator and receives zero.

Build the formal 163-workload denominator:

```bash
uv run sol-execbench solar release-build out/release \
  --backend torchview_extended_einsum \
  --orojenesis-home /path/to/reviewed/orojenesis
```

Each workload record binds the selected fixed IR path, operator graph, dynamic
IR graph name, conversion attestation, formal analysis, and request manifest.
Use a separate release root with `--backend make_fx_aten`; a release index
cannot mix or resume with a different path. Verification rejects
missing or duplicate workloads, diagnostic bounds, wrong
reference/architecture identity, untrusted Orojenesis policy, and artifact
hash drift.

## Build statements and bundle

Create run statements after their complete traces verify:

```bash
uv run sol-execbench score build-statement \
  out/release/baseline/plan.json
uv run sol-execbench score build-statement \
  out/release/candidate/plan.json
```

The SOLAR release builder writes `statements/solar.json`. Assemble the three
statement references:

```bash
uv run sol-execbench score assemble-bundle out/release
uv run sol-execbench score official out/release/release-bundle.json
```

`release-bundle.json` directly contains SHA-256 and byte-size references for:

```text
corpus/manifest.yaml
statements/baseline.json
statements/candidate.json
statements/solar.json
```

The verifier requires the repository-pinned corpus hash, corpus-pinned baseline
ID, exact canonical baseline implementation, one source revision, identical
validated runtime environments, full problem/workload coverage, passing
baseline traces, complete candidate traces, and the exact SOLAR artifact
denominator. Raw caller-supplied timing values are not accepted.

## Score semantics

For every correct workload:

```text
S(T_k) = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))
```

`T_b` is the canonical baseline run in the release bundle. Incorrect candidates
score zero. Correct candidates must have finite positive runtimes and satisfy
`T_b > T_SOL` and `T_k >= T_SOL`; violations are audit failures rather than
values to clip. Workloads are averaged within each problem, then the 43 problem
means receive equal weight.
