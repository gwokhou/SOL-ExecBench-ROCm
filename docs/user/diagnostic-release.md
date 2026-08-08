# Diagnostic release packaging

Performance diagnostics are diagnostic-only. They never change canonical Trace
timing, `T_SOL`, SOL Score, leaderboard values, or rewards. A diagnostic release
object packages a **verified** compact publication projection into a
byte-reproducible archive with a typed attestation; it never authorizes an
official score or leaderboard result.

## The release object

The governed packager consumes one verified `publication.json` and emits three
things:

- `NAME.tar.zst` — a deterministic zstd archive of the exact publication tree.
  `tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --zstd`
  makes the archive byte-for-byte reproducible; no timestamps, owners, or
  directory order leak into the bytes.
- `NAME.attestation.json` — a `sol_execbench.diagnostic_release_attestation.v2`
  object that binds the release identity, publication digest, archive digest and
  size, the exact sorted inventory digest, case count, source revision, and
  producer version.
- An immutable `DiagnosticReleaseLifecycleManifest` under
  `data/store/releases/<release_id>/manifest.json` when a store root is
  supplied, recording the archive and attestation digests with retention class
  `publication_release`.

Packaging is governed: the full publication tree is re-verified (inventory,
hashes, symlinks, case identity, reproducible inference) before any archive is
written, and the command refuses to overwrite an existing archive or release
object.

```bash
sol-execbench --format json diagnostics release package \
  --manifest data/publications/microarchitecture-diagnostics-v7-cycle3/publication.json \
  --archive-output data/publications/microarchitecture-diagnostics-v7-cycle3.tar.zst \
  --attestation-output data/publications/microarchitecture-diagnostics-v7-cycle3.attestation.json \
  --source-revision 19f195a8
```

## Blob identity

Promoted source corpora reference their case artifacts by content-addressed
blob keys (`sha256`), not by path. The lifecycle blob store lives under
`data/store/blobs/sha256/<digest>` by default and is overridable with the
`SOL_EXECBENCH_DIAGNOSTIC_STORE` environment variable. Every read re-verifies
the stored content against its key, and a promoted corpus depends on no
historical physical path tree. Compact publications remain self-contained:
their projected corpora use tree-backed references resolved relative to the
publication root.

## Verifying a downloaded archive

After download, verify the external archive SHA-256 and the unpacked
publication before consuming `development.json`, `calibration/profile.json`, or
`inference.json`:

```bash
sol-execbench --format json diagnostics release verify \
  --archive microarchitecture-diagnostics-v7-cycle3.tar.zst \
  --expected-sha256 <published sha256>
```

The verifier recomputes the archive digest, unpacks it, and runs
`verify-publication-projection` on the unpacked tree. It is CPU-only and needs
only the archive; the large process roots are not distribution dependencies.

## Publishing to GitHub

Release publishing uses a draft-first pattern. The operator attaches the
archive and attestation to a draft GitHub Release bound to a source tag:

```bash
gh release create <tag> \
  microarchitecture-diagnostics-v7-cycle3.tar.zst \
  microarchitecture-diagnostics-v7-cycle3.attestation.json \
  --draft --target <source revision>
```

A GitHub-hosted `diagnostic-release.yml` job then takes the tag via
`workflow_dispatch`, checks out that revision, downloads the draft release
assets, recomputes the archive SHA-256 and compares it to the attestation,
verifies the unpacked publication, and requires a byte-identical deterministic
rebuild. Only after all of that succeeds does it mark the draft release
published. A human reviews the draft before the final `gh release edit` runs, or
the workflow is re-dispatched later.

The self-hosted GPU runner remains a collection producer and never receives
durable `contents: write` release authority. Only the GitHub-hosted release job
scopes `contents: write`, and only after tag/revision and checksum
verification.
