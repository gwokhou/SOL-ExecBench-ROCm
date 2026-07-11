# Prerelease Artifact Bundle

This guide defines the versioned artifact bundle used for the public
engineering prerelease and research preview. The bundle is a review package. It
does not change benchmark schemas, scoring semantics, trace authority, paper
parity, leaderboard readiness, hard-sandbox status, or hardware-validation
claims.

## Generate A Bundle

From a clean checkout or release tag:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/build_prerelease_artifact_bundle.py \
  --version v1.26.0-rc1 \
  --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1
```

The default command runs CPU-safe release-candidate validation and attempts to
capture `sol-execbench doctor --json` as diagnostic environment evidence. The
doctor command is optional evidence; if it is unavailable, the bundle records
that status instead of treating it as paper-scale validation failure.

For a dry evidence package that records the missing validation gap explicitly:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/build_prerelease_artifact_bundle.py \
  --version v1.26.0-rc1 \
  --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1 \
  --skip-release-validation
```

Do not publish a release candidate from a skipped-validation bundle without
running the normal validation path first.

## Release Baseline Evidence

When publishing a scored release, attach the generated complete-suite baseline
bundle and its independent rerun verification together:

```bash
uv run scripts/internal/release/build_prerelease_artifact_bundle.py \
  --version v2.14.0-rc1 \
  --output-dir out/prerelease_artifact_bundle/v2.14.0-rc1 \
  --release-baseline-bundle out/release/release-baseline.json \
  --release-baseline-verification out/release/release-baseline-verification.json
```

The bundle copies both files under `release_baseline/`, records their checksums,
and reports the complete workload denominator as `official`, `derived`, and
`blocked`.  A derived or blocked workload remains publishable review evidence,
but does not make a full-suite official claim.

## Output Layout

The bundle writes:

- A machine-readable manifest in the selected output directory.
- A reviewer summary Markdown file in the selected output directory.
- `SHA256SUMS`: checksums for generated bundle files.
- `release_candidate_validation/`: release validation JSON and Markdown.
- `transcripts/`: command transcripts with redacted log tails.
- `environment/`: diagnostic environment evidence when available.

The manifest also records checksums for referenced source documents such as
`docs/CLAIMS.md`. Bundle commands stream stdout and stderr to temporary files
and retain only bounded redacted transcript tails in the manifest. `SHA256SUMS`
is written in deterministic path order; missing file digests may be computed in
parallel within a run, but the output ordering remains stable.

The bundle must not contain NVIDIA SOL-ExecBench original dataset content or
ROCm-migrated derivatives of that content. The prerelease readiness checker
scans bundle paths using the dataset policy in `provenance.toml` and blocks
release bundles that include restricted NVIDIA dataset payloads. FlashInfer
Trace content remains a separate Apache-2.0 source boundary and must retain the
required license and attribution notices when redistributed.

## Authority Classes

Every artifact or evidence surface is mapped to one of these classes:

| Class | Meaning |
| --- | --- |
| `canonical` | The canonical benchmark artifact class. Trace JSONL remains canonical; the bundle does not generate traces by default. |
| `diagnostic-only` | Review evidence or sidecar diagnostics only; not correctness, timing, score, paper-parity, leaderboard, or hardware-validation authority. |
| `provisional` | Scoped prerelease evidence such as bounded dataset slices or support wording; useful for review but not paper parity. |
| `deferred` | Known required evidence that is intentionally out of scope for this prerelease package. |
| `unavailable` | Evidence that cannot currently be collected, such as CDNA4 validation while suitable hardware is inaccessible. |

## Claim Boundary

The bundle is engineering prerelease and research preview evidence only. It is
not full 235-problem paper validation, upstream SOLAR parity, leaderboard
readiness, hard-sandbox evidence, native-host validation inferred from Docker,
full validation of the MI300X GPU model under CDNA3, or CDNA4 validation.

MI300X and MI308X are sibling GPU products under the CDNA3 architecture family
and share the `gfx942` code path. Current CDNA3/gfx942 validation
infrastructure evidence was recorded on MI308X, not MI300X, so full-suite
MI300X validation remains blocked until timeout, clock-lock, timing, score,
FP8, low-precision, and exact-hardware evidence boundaries are resolved.
CDNA4 validation is unavailable because suitable hardware is not currently
accessible.
