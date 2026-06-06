# Prerelease Readiness Gates

This guide defines the CPU-safe release-readiness gate for the public
engineering prerelease and research preview. The gate consumes a generated
prerelease artifact bundle and fails publication checks when required evidence
is missing, checksums drift, claim boundaries regress, or known gaps use an
unreviewed status.

## Run The Gate

Generate the bundle first:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/build_prerelease_artifact_bundle.py \
  --version v1.26.0-rc1 \
  --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1
```

Then run readiness:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/check_prerelease_readiness.py \
  --bundle-dir out/prerelease_artifact_bundle/v1.26.0-rc1 \
  --output-dir out/prerelease_readiness/v1.26.0-rc1
```

The gate writes:

- `prerelease_readiness.json`
- `prerelease_readiness.md`

## Blocking Conditions

The gate exits nonzero when it finds:

- missing `prerelease_artifact_bundle.json` or `SHA256SUMS`
- missing required artifact files
- checksum mismatches
- unknown authority classes
- missing required authority classes
- truthy forbidden claim-boundary fields
- missing claim-boundary documents or representative non-claim phrases
- known gaps with statuses outside `blocking`, `deferred`, `unavailable`, or
  `diagnostic-only`

## Known Gap Review

Known gaps must be visible before publishing. The gate reports each known gap
with its status:

| Status | Meaning |
| --- | --- |
| `blocking` | Must be fixed before publishing. |
| `deferred` | Explicitly outside this prerelease package. |
| `unavailable` | Cannot currently be collected, such as CDNA4 validation while suitable hardware is inaccessible. |
| `diagnostic-only` | Review context only; not release authority. |

MI300X remains a distinct CDNA3 `gfx942` hardware-validation target. Current
CDNA3/gfx942 validation infrastructure evidence was recorded on MI308X, not
MI300X, so full MI300X validation remains blocked until timeout, clock-lock,
timing, score, FP8, low-precision, and exact-hardware evidence boundaries are
resolved. CDNA4 validation is unavailable because suitable hardware is not
currently accessible.
