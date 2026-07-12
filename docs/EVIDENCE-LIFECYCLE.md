# GitHub Evidence Lifecycle

Published evidence has a Git-tracked lifecycle index at
`docs/releases/evidence-lifecycle.json`. It is the authoritative list of active,
superseded, and revoked authority slices; GitHub Releases are immutable distribution
objects, not the source of lifecycle state.

## Permission boundary

Publishing is deliberately split into two workflows:

- `Prepare evidence bundle` runs only on a repository-restricted self-hosted runner
  labelled `evidence-producer`. It has `contents: read` only and stages raw local
  evidence from `$RUNNER_TEMP/evidence/<release>/source` into a short-lived Actions
  artifact. It has no Git, release, attestation, or OIDC write permission.
- `Publish evidence bundle` downloads only a successful `Prepare evidence bundle`
  artifact from `main`, validates its manifest and archive checksum, creates a new
  release only if that tag does not already exist, attests the archive, downloads it
  again from GitHub, and verifies the extracted closure. Only this job receives
  `contents: write`, `attestations: write`, and `id-token: write`.

## Single-maintainer mode

This repository currently has one maintainer. It intentionally does **not** claim
independent human approval: `evidence-publish` and `evidence-lifecycle` are GitHub
Environments restricted to `main`, without required reviewers. The publish and
revoke workflows are manual-only and also require `github.ref` to be `main`.

This mode still limits write access to the two narrowly scoped jobs and keeps all
other workflows read-only, but it is not dual control. Add a distinct reviewer or
team and reinstate required-reviewer protection before making a two-person approval
claim.

Configure the repository before dispatching either workflow:

1. Create `evidence-publish` and `evidence-lifecycle` GitHub Environments and
   restrict their deployment branch policies to `main`.
2. Restrict the `evidence-producer` self-hosted runner group to this repository and
   do not store long-lived GitHub tokens, signing keys, or production credentials on
   it. The runner only needs access to the locally generated evidence input.
3. Permit the publisher and lifecycle jobs to push only the lifecycle
   index. Do not grant `contents: write` to pull-request, scheduled, or ordinary CI
   workflows.

The code-quality workflow remains read-only. Neither `pull_request` nor
`pull_request_target` can publish, revoke, or mutate evidence.

## State machine

The executable state machine is in `tools/evidence_lifecycle.py` and validates every
commit to the lifecycle index.

| State | Meaning | Allowed transition |
| --- | --- | --- |
| `published` | Active evidence whose public archive passed round-trip verification. | `published → superseded` by a protected successor publication, or `published → revoked` by the protected revoke workflow. |
| `superseded` | Historic but valid evidence replaced by a later slice. | Terminal; artifacts remain downloadable and auditable. |
| `revoked` | Historic evidence that must not be used. | Terminal; a public reason is required and assets are retained for audit. |

Every Monday, `Verify published evidence` downloads every `published` archive,
checks the archive SHA-256 stored in the lifecycle index, extracts it, and invokes
`baseline publication verify`. A failed run is a release-integrity incident and must
be handled before making new authority claims.

## Operator sequence

1. Merge the manifest under `docs/releases/<release>.evidence.json`.
2. Place the source closure on the restricted producer runner at
   `$RUNNER_TEMP/evidence/<release>/source`.
3. Dispatch `Prepare evidence bundle` on `main` and retain its successful run ID.
4. Dispatch `Publish evidence bundle` on `main` with that run ID. If a run stops
   after creating the immutable Release but before recording its
   lifecycle commit, redispatch the same inputs. The workflow re-verifies the
   public archive, attests it, and records it without replacing release assets.
5. Confirm the lifecycle commit and the scheduled verifier are green. To replace an
   older release, pass its name as `supersedes`; to revoke a release, use the
   protected revoke workflow rather than deleting the GitHub Release.
