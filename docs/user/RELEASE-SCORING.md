# Release and Official Score Workflow

Canonical Trace JSONL is the execution artifact. A numeric formula result is
official only when it is derived from the repository-pinned corpus and a
publisher-authored, content-addressed release bundle.

## v2 scoring contract

The schema v7 RX 9060 XT corpus contains 43 scored problems and 163 scored
workloads. The FP8 compatibility sentinel and the provenance-retained
`l2n55_matmul_maxpool_sum_scale` target-incompatible problem are excluded from
the denominator. The latter keeps its original AKA shapes; it is excluded
because its trusted-reference IPC payload exceeds the bounded protocol.

The manifest authorizes official scoring. A publisher release bundle must
still be distributed under `RELEASE/` before a numeric result is official:

```yaml
official_scoring:
  status: available
  baseline_id: rx9060xt-gfx1200-reference-v2
  release_policy: content_addressed_publisher_v1
  required_evidence:
    - content_addressed_release_baseline
    - content_addressed_candidate_execution
    - pinned_solar_manifests
```

The release policy follows the paper's release-defined baseline `T_b`,
candidate runtime `T_k`, formal bound `T_SOL`, and controlled harness. It does
not require multiple release roles, detached signatures, or a second baseline
run.

Content hashes prove that bundle artifacts have not changed. They do not prove
who published a bundle. Publisher authenticity therefore comes from the
repository or release channel that distributes the bundle.

## Publication cutover

The pending manifest cannot be promoted in place into a final release. Baseline
plans and statements bind the exact corpus-manifest hash, while authorizing the
v2 policy changes that manifest and the repository-owned
`OFFICIAL_CORPUS_MANIFEST_SHA256` pin. Any baseline or SOLAR statement built
against the pending manifest is reproducibility source evidence only and must
be rebuilt for the publication revision.

The current `content_addressed_publisher_v1` contract is candidate-specific. A
complete `release-bundle.json` always binds these three evidence classes:

- `content_addressed_release_baseline`;
- `content_addressed_candidate_execution`;
- `pinned_solar_manifests`.

A publisher cutover therefore has the following atomic contract:

1. Select the concrete candidate to score and prepare the final corpus manifest
   with `status: available`, release policy
   `content_addressed_publisher_v1`, the canonical baseline ID, and the three
   required-evidence values above.
2. Update `OFFICIAL_CORPUS_MANIFEST_SHA256` and all pending-state tests and docs,
   then establish one clean source revision containing that exact policy.
3. Build and run baseline, candidate, and SOLAR evidence from that clean
   revision against the final manifest. All three statements must bind the same
   source revision and validated runtime environment where required.
4. Assemble and verify the complete bundle, then package it into a
   deterministic, self-verifying release archive:

   ```bash
   uv run sol-execbench score release-package out/release/release-bundle.json \
     --archive-output score-release.tar.zst \
     --attestation-output attestation.json \
     --source-revision SOURCE_GIT_SHA
   ```

   `release-package` first re-runs the fail-closed verifier, then collects the
   exact transitively-referenced evidence set (the same files the verifier
   checks — execution plans and trace sidecars are excluded by construction) and
   writes a byte-reproducible zstd archive plus a content-addressed attestation
   binding the bundle, archive, inventory, source revision, and reproduced
   official score. Upload the archive and attestation as a **draft** GitHub
   Release, then trigger the `Score Release` workflow
   (`.github/workflows/score-release.yml`): it downloads the draft, verifies the
   archive SHA-256 against the attestation, reproduces the official score with
   `score release-verify`, requires a byte-identical deterministic rebuild, and
   promotes the draft to published. The GitHub-hosted workflow job is the only
   component that holds release authority; the self-hosted GPU runner only
   produces evidence.
5. Verify that `sol-execbench score status` reports both policy authorization
   and the published release. Anyone can reproduce the official score from the
   distributed archive:

   ```bash
   uv run sol-execbench score release-verify score-release.tar.zst \
     --expected-sha256 "$(python -c 'import json;print(json.load(open("attestation.json"))["archive"]["sha256"])')"
   ```

A baseline-and-SOLAR-only directory is not a complete official-score release.
Publishing a reusable baseline independently would require a separate reviewed
contract; it is not represented by the current official-score API.

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

`--jobs` defaults to `1`, preserving the serial release path. Values above one
serialize graph extraction, conversion, and replay on the selected GPU while
overlapping CPU-heavy formal analysis across workloads. Each mapper uses every
logical CPU visible in the current process affinity; for example, the default
policy selects 32 threads on a 16-core/32-thread host and 16 threads when the
same process is affinity-limited to 16 logical CPUs. The builder computes the
outer-job limit in the same logical-CPU units. Under the default full-CPU mapper
policy this retains one outer job; values above one require a future explicit
per-worker CPU partition rather than oversubscribing the same affinity set. If
logical CPU availability cannot be detected, the mapper conservatively uses one
thread and only one outer job is accepted.

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
