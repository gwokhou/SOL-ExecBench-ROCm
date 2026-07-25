# Release and Official Score Workflow

Canonical Trace JSONL is the execution artifact. A numeric formula result is
official only after every input crosses an independently verifiable release
authority boundary.

## Current v3 status

The fixed RX 9060 XT denominator contains 35 scored problems and 122 scored
workloads. The FP8 compatibility sentinel and the provenance-retained
`l2n55_matmul_maxpool_sum_scale` target-incompatible problem are excluded. The
upstream problem's `32768 x 32768` FP32 weight alone exceeds the bounded trusted
reference IPC payload, so the release does not silently resize it or weaken the
IPC limit. Corpus selection derives contiguous input/output storage directly
from Definition shapes, workload axes, and dtype widths. A schema-derived case
above the IPC limit is rejected at the static stage with
`reference_ipc_payload_limit` and byte-count metrics; no reference worker, GPU
allocation, or live probe is started. Cases not proven incompatible remain
subject to the live probe.

The checked-in corpus still records `official_scoring.status: unavailable` and
`reason_code: release_authority_not_published`.

The implementation is no longer the blocker: the repository provides a
deterministic baseline planner/runner, full-corpus SOLAR release builder,
detached Ed25519 statement verification, exact bundle verification, and the
official scorer. The current release remains unavailable because no four-role
public-key policy or signed evidence set has been published. Caller-authored
measurement, baseline, or SOLAR JSON cannot become authority.

## Reproduce the formal mapper

Before generating formal manifests, build the pinned mapper twice from the
digest-pinned builder and Ubuntu snapshot:

```bash
scripts/internal/orojenesis/verify_reproducible_build.sh \
  out/orojenesis-reproducible
```

The command uses two clean builds and publishes the first artifact only when
the mapper and provenance are byte-identical. A reviewer must independently
inspect the printed digest and provenance before adding that digest to the
repository-owned `OROJENESIS_TRUSTED_MAPPER_SHA256` allowlist. A locally
self-declared provenance file is never sufficient.

## Release workspace

First publish the four independently administered Ed25519 public keys in the
corpus manifest. Private keys must never enter the repository or release
workspace. Then create the content-addressed trusted-reference baseline:

```bash
uv run sol-execbench baseline release-build out/release \
  --baseline-id rx9060xt-gfx1200-v1 \
  --source-revision SOURCE_GIT_SHA
```

The release-defined v1 baseline executes the exact corpus-pinned eager PyTorch
reference. This keeps correctness and provenance invariant across all 122
workloads; compiler rewrites are candidate implementations, not hidden changes
to the scoring anchor. This is not a claim that an unpublished agent-frontier
solution set exists. `SOURCE_GIT_SHA` must be the current clean commit: the
release runner and SOLAR builder inspect all release-relevant tracked and
untracked source paths and fail if the mounted tree differs.

Run both plans inside the hardened container on the exact pinned GPU:

```bash
./scripts/run_docker.sh -- sol-execbench baseline release-run \
  /outputs/release/baseline/plan.json
./scripts/run_docker.sh -- sol-execbench baseline release-run \
  /outputs/release/rerun/plan.json
```

The wrapper resolves the immutable local Docker image ID and passes it into the
container. Each run records that `sha256:` ID together with the clean source
revision in its signed environment artifact. Baseline, rerun, and candidate
must use the same image identity. The verifier also rejects a rerun that reuses
any baseline trace artifact verbatim.

Candidate inputs use one `solution.json` under every scored problem path:

```bash
uv run sol-execbench baseline candidate-build out/release CANDIDATE_ROOT \
  --candidate-id CANDIDATE_ID \
  --source-revision SOURCE_GIT_SHA
./scripts/run_docker.sh -- sol-execbench baseline release-run \
  /outputs/release/candidate/plan.json
```

Build and verify the formal 122-workload denominator:

```bash
uv run sol-execbench solar release-build out/release \
  --orojenesis-home /path/to/reviewed/orojenesis
```

Each output contains the operator graph, einsum graph, conversion attestation,
formal analysis, and request manifest. The index builder rejects missing or
duplicate workloads, diagnostic bounds, wrong reference/architecture identity,
untrusted Orojenesis policy, and artifact hash drift.

## Statements, signatures, and bundle

Create unsigned run statements only after their complete traces verify:

```bash
uv run sol-execbench score build-statement out/release/baseline/plan.json
uv run sol-execbench score build-statement out/release/rerun/plan.json
uv run sol-execbench score build-statement out/release/candidate/plan.json
```

The SOLAR release builder writes `statements/solar.json`. Each independent
authority signs exactly its role payload with Ed25519, producing:

```text
signatures/baseline.sig
signatures/rerun.sig
signatures/candidate.sig
signatures/solar.sig
```

Release administration may use OpenSSL, for example:

```bash
openssl pkeyutl -sign -inkey ROLE_PRIVATE_KEY.pem -rawin \
  -in out/release/statements/ROLE.json \
  -out out/release/signatures/ROLE.sig
```

Assemble and score only after all four signatures verify against the
repository-pinned public keys:

```bash
uv run sol-execbench score assemble-bundle out/release
uv run sol-execbench score official out/release/release-bundle.json
```

The verifier requires four distinct keys, the exact immutable corpus, a
corpus-pinned baseline ID, one source revision, identical validated runtime
environment identities (including immutable container image and committed
source), passing baseline/rerun coverage, distinct rerun traces, exact
implementation reuse in the rerun, acceptable rerun drift, trusted candidate
traces, and all formal SOLAR artifacts.

## Score semantics

For each workload, the baseline runtime is the arithmetic mean of the original
and independent-rerun measurements. Incorrect candidates score zero. Correct
candidates must satisfy finite positive runtimes, `T_b > T_SOL`, and
`T_k >= T_SOL`; violations are audit failures rather than values to clip.
Workloads are averaged within each problem, then the 35 problem means receive
equal weight.
