# Public Prerelease Publishing Guide

This guide prepares the v1.26 public engineering prerelease and research
preview. It does not publish a release by itself; it defines the maintainer
checklist and links required for a GitHub prerelease or equivalent public
release page.

## Pre-Publication Checklist

1. Confirm the tree and tag context.

```bash
git status --short
git log --oneline -5
```

2. Generate the artifact bundle.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/build_prerelease_artifact_bundle.py \
  --version v1.26.0-rc1 \
  --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1
```

3. Run the readiness gate.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/check_prerelease_readiness.py \
  --bundle-dir out/prerelease_artifact_bundle/v1.26.0-rc1 \
  --output-dir out/prerelease_readiness/v1.26.0-rc1
```

4. Attach or link release assets.

- prerelease artifact bundle JSON report
- prerelease artifact bundle Markdown summary
- SHA-256 checksum manifest
- prerelease readiness JSON report
- prerelease readiness Markdown summary

5. Fill the release draft in `docs/releases/v1_26_prerelease_draft.md`.

## Required Public Links

Public materials must link:

- `docs/user/provenance.md`
- `docs/user/compliance.md`
- `docs/internal/prerelease_artifact_bundle.md`
- `docs/internal/prerelease_readiness.md`
- `docs/user/research_preview.md`
- `docs/user/rocm.md`
- `docs/user/CLAIMS.md`
- `docs/user/GETTING-STARTED.md`
- `docs/user/rocm_timing.md`
- `docs/user/RESEARCHER-GUIDE.md`
- `docs/internal/v1_25_release_notes.md`
- `docs/internal/v1_25_prerelease_checklist.md`

The v1.25 links are historical engineering-prerelease context that the v1.26
public review package still depends on. They are not the Python package
version.

## Required Wording

Use this framing:

> v1.26 is an engineering prerelease and research preview for the SOL ExecBench
> ROCm port.

Also state that this repository is an Apache-2.0 ROCm port of NVIDIA
SOL-ExecBench. Retained or derivative upstream files preserve applicable
NVIDIA notices; independent ROCm work uses project attribution. The
SOL-ExecBench paper is the benchmark and methodology citation, not a
file-level copyright assignment for independent ROCm work.

Do not present it as a stable benchmark authority release. Do not claim full
235-problem paper-scale validation, upstream SOLAR parity, hosted leaderboard
readiness, hard-sandbox authority, native-host validation inferred from Docker,
completed full validation of the MI300X GPU model under CDNA3, or CDNA4
validation. Do not imply NVIDIA or AMD endorsement.

MI300X and MI308X are sibling GPU products under the CDNA3 architecture family
and share the `gfx942` code path. Current CDNA3/gfx942 validation
infrastructure evidence was recorded on MI308X, not MI300X, so full MI300X
validation remains blocked until timeout, clock-lock, timing, score, FP8,
low-precision, and exact-hardware evidence boundaries are resolved. CDNA4
validation is unavailable because suitable hardware is not currently
accessible.

## Known Limitations To Include

- Trace JSONL remains canonical; sidecars are diagnostic-only unless explicitly
  narrower.
- Bounded dataset slices are provisional research-preview evidence, not paper
  parity.
- Docker/container ROCm user-space evidence is not native-host validation.
- No hosted leaderboard or remote submission service is included.
- No hardened multi-tenant sandbox is included.
