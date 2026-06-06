# v1.26.0-rc1: Engineering Prerelease And Research Preview

> Replace asset placeholders before publishing.

v1.26.0-rc1 is an engineering prerelease and research preview for the SOL
ExecBench ROCm port. It packages versioned validation artifacts, release
readiness gates, and research-preview documentation so reviewers can inspect
the current ROCm evidence without treating it as a stable benchmark authority
release.

## Assets

- Artifact bundle JSON: `<attach prerelease_artifact_bundle.json>`
- Artifact bundle Markdown: `<attach prerelease_artifact_bundle.md>`
- SHA-256 checksums: `<attach SHA256SUMS>`
- Readiness JSON: `<attach prerelease_readiness.json>`
- Readiness Markdown: `<attach prerelease_readiness.md>`

## What This Prerelease Contains

- Provenance and compliance policy:
  `docs/provenance.md`, `docs/compliance.md`.
- A versioned prerelease artifact bundle workflow:
  `docs/prerelease_artifact_bundle.md`.
- A prerelease readiness gate:
  `docs/prerelease_readiness.md`.
- A bounded research preview evidence package:
  `docs/research_preview.md`.
- ROCm support and hardware boundary notes:
  `docs/rocm.md`.
- Claim-boundary policy:
  `docs/CLAIMS.md`.
- First-run setup and troubleshooting:
  `docs/GETTING-STARTED.md`.
- Timing semantics and profiler evidence boundaries:
  `docs/rocm_timing.md`.
- Research workflows:
  `docs/RESEARCHER-GUIDE.md`.

## Validation

Recommended maintainer commands before publishing:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/build_prerelease_artifact_bundle.py \
  --version v1.26.0-rc1 \
  --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1

UV_CACHE_DIR=/tmp/uv-cache uv run scripts/check_prerelease_readiness.py \
  --bundle-dir out/prerelease_artifact_bundle/v1.26.0-rc1 \
  --output-dir out/prerelease_readiness/v1.26.0-rc1
```

The attached bundle should include checksums, command transcripts, release
validation output, environment evidence, known gaps, and authority-class
mappings. The readiness report should be reviewed for blocking findings before
publishing.

## Attribution

This prerelease is for the Apache-2.0 SOL ExecBench ROCm port. The upstream
project is NVIDIA SOL-ExecBench. Retained or derivative upstream files preserve
applicable NVIDIA notices, and independent ROCm work uses project attribution.
The SOL-ExecBench paper is cited for benchmark and methodology context; it is
not a file-level copyright assignment for independent ROCm work.

This prerelease does not imply NVIDIA or AMD endorsement.

## Research Preview Scope

Trace JSONL remains the canonical run artifact. Release validation, readiness,
environment, profile, static, Matrix, closure, consistency, claim-upgrade, and
trust-summary outputs are diagnostic-only sidecar evidence unless a narrower
document gives them a more limited role. Bounded dataset slices are provisional
research-preview evidence, not full paper validation.

AMD-native SOL and score reports are local derived evidence from ROCm traces
and AMD-side bound artifacts. They are not upstream SOLAR parity, NVIDIA B200
equivalence, or official leaderboard equivalence.

## Support Boundaries

- RDNA4 evidence is engineering-prerelease evidence only where archived
  artifacts and commands support the recorded scope.
- Docker/container ROCm user-space evidence is not native-host validation.
- MI300X remains a distinct CDNA3 `gfx942` hardware-validation target. Current
  CDNA3/gfx942 validation infrastructure evidence was recorded on MI308X, not
  MI300X, so full MI300X validation remains blocked until timeout, clock-lock,
  timing, score, FP8, low-precision, and exact-hardware evidence boundaries are
  resolved.
- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.

## Not Claimed

This prerelease does not claim:

- full 235-problem paper-scale validation
- upstream SOLAR parity
- hosted leaderboard readiness
- stable benchmark authority release status
- hard multi-tenant sandbox authority
- native-host validation inferred from Docker/container evidence
- completed full MI300X validation on CDNA3 `gfx942`
- CDNA4 validation

## More Links

- v1.25 release notes: `docs/v1_25_release_notes.md`
- Prerelease checklist: `docs/v1_25_prerelease_checklist.md`
- Public publishing checklist: `docs/public_prerelease.md`
- Provenance policy: `docs/provenance.md`
- Compliance notes: `docs/compliance.md`
