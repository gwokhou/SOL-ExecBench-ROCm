# Evidence Publication

Raw benchmark output, profiler traces, and downloaded datasets do not belong in
Git.  A score claim is nevertheless reproducible only when Git fixes the exact
identity of every externally stored input.  This repository uses
`sol_execbench.evidence_publication_manifest.v1` for that purpose.

## Current TODO: publish the authority slice

**Status: blocked on an available immutable public artifact store.** The local
`out/authority-release-20260711-v3/` directory must not be committed to Git.
When a suitable free public store is available, upload that directory as one
versioned bundle, create and commit its publication manifest, and run
`baseline publication-verify` from a clean download. Until then, all claims
about this slice must use **locally verified authority slice** and must not use
**published authority slice**.

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
   into a clean release directory.
2. Upload that directory without changing filenames.  Do not use a mutable
   `latest` URL.
3. Create the manifest using the final SHA-256 values, candidate solution hash,
   source commit, container image digest, and release URL.
4. Generate official-score evidence with content-addressed candidate solution,
   trace, and timing evidence. Its environment fingerprint, clock policy, and
   timing policy must match the release baseline.
5. Commit the manifest and tag the same Git revision as the evidence release.
6. Download the bundle into a fresh directory and verify it:

   ```bash
   uv run sol-execbench baseline publication-verify \
     --manifest docs/releases/<release>.evidence.json \
     --artifact-root /path/to/downloaded-release
   ```

7. Record that verification in CI or the release notes.  A checksum failure,
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
