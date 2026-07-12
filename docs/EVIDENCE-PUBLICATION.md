# Evidence Publication

Raw benchmark output, profiler traces, and downloaded datasets do not belong in
Git.  A score claim is nevertheless reproducible only when Git fixes the exact
identity of every externally stored input.  This repository uses
`sol_execbench.evidence_publication_manifest.v1` for that purpose.

## Current published authority slice

**Status: published.** Legacy v3 evidence was rejected by the current official gate and has
been removed from the local worktree; it must not be restored or uploaded as an
authority release. Its replacement is the local
`gfx1200-20260712-hipblaslt-v4` source evidence tree at
`out/authority-release-20260712-v3-closure/` (the directory name is historical).
It contains 25 v4 bounds, matching fusion validation, baseline/bundle/rerun evidence,
candidate identity, and official-score evidence, plus retained local generation
diagnostics. It is not itself the upload directory. Stage exactly the manifest-listed
files into a new directory, then verify that exact directory:

```bash
uv run sol-execbench baseline publication stage \
  --manifest docs/releases/gfx1200-20260712-hipblaslt-v4.evidence.json \
  --source-root out/authority-release-20260712-v3-closure \
  --destination out/gfx1200-20260712-hipblaslt-v4-publish

uv run sol-execbench baseline publication verify \
  --manifest docs/releases/gfx1200-20260712-hipblaslt-v4.evidence.json \
  --artifact-root out/gfx1200-20260712-hipblaslt-v4-publish
```

The immutable bundle is now published as
[`gfx1200-20260712-hipblaslt-v4-evidence.tar.gz`](https://github.com/gwokhou/SOL-ExecBench-ROCm/releases/download/gfx1200-20260712-hipblaslt-v4/gfx1200-20260712-hipblaslt-v4-evidence.tar.gz)
in the [GitHub Release](https://github.com/gwokhou/SOL-ExecBench-ROCm/releases/tag/gfx1200-20260712-hipblaslt-v4).
Its SHA-256 is
`0866ec026f53d6e77d73258c24f19108dce746c1a463dc6536d9e641a758513b`.
The bundle was downloaded into a fresh temporary directory, extracted, and verified
against the Git-tracked manifest with `baseline publication verify` on 2026-07-12.
This is a **published authority slice**, not a full-suite or leaderboard claim.

Lifecycle policy, protected workflow setup, and the operator sequence are in
[`docs/EVIDENCE-LIFECYCLE.md`](EVIDENCE-LIFECYCLE.md).

## What is committed

Commit the small JSON manifest next to the release notes.  It must contain:

- the release name and explicit scope;
- source repository and immutable Git revision;
- container image digest;
- HTTPS base URI for an immutable evidence bundle;
- the candidate solution and its candidate trace, each with SHA-256;
- the timing-evidence artifact cited by the candidate trace, with SHA-256;
- every artifact needed to reproduce or verify the stated score, each with a
  safe relative path and SHA-256.

The manifest is intentionally a complete authority contract, not an arbitrary
checksum list. It requires unique entries for the candidate solution, candidate
trace, candidate timing evidence, scoring baseline, release-baseline bundle,
rerun verification, suite manifest, and official-score evidence. It must also
include every checksum cited by an official workload's trace, bound, and
hardware-model inputs. Verification checks that the baseline, suite manifest,
bundle, rerun, candidate identity, scope, and official-score evidence all
describe the same release.

Use [the example manifest](examples/evidence-publication-manifest.v1.example.json)
as the starting point.  A manifest is an index, not a substitute for its
artifacts: the referenced bundle must be uploaded to a versioned GitHub Release
or other immutable HTTPS object store before publishing an authority claim.

## Publisher procedure

1. Generate baseline, bound, rerun, candidate trace, and official-score files
   into a clean release directory. Copy or regenerate every file referenced from
   another release directory (including suite manifest, candidate solution, and
   hardware model), then rewrite references to paths inside this closure.
2. Create the manifest using the final SHA-256 values, candidate solution hash,
   source commit, container image digest, and immutable release URL.
3. Use `baseline publication stage` to create a separate, exact upload directory.
   The command copies only manifest-listed files and re-verifies every authority
   reference; it rejects undeclared files in the staged result.
4. Upload that staged directory without changing filenames.  Do not use a mutable
   `latest` URL.
5. Generate official-score evidence with content-addressed candidate solution,
   trace, and timing evidence. Its environment fingerprint, clock policy, and
   timing policy must match the release baseline.
6. Commit the manifest and tag the same Git revision as the evidence release.
7. Download the bundle into a fresh directory with no sibling `out/` release
   directories, and verify it:

   ```bash
   uv run sol-execbench baseline publication verify \
     --manifest docs/releases/<release>.evidence.json \
     --artifact-root /path/to/downloaded-release
   ```

8. Record that verification in CI or the release notes.  A checksum failure,
   missing candidate identity, changed source revision, or unresolved artifact
   blocks a *published* authority claim.

## Claim boundary

The existing `official_score_evidence.v1` gate establishes that a locally
available score has consistent baseline, bound, and rerun inputs.  It does not
by itself publish those files.  Use the phrase **locally verified authority
slice** until a matching publication manifest has passed the command above.
Only then call the result a **published authority slice**.  Neither term means
full-suite, NVIDIA/SOLAR-parity, or leaderboard authority unless that broader
scope is separately published and verified.
