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
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/build_prerelease_artifact_bundle.py \
  --version v1.26.0-rc1 \
  --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1
```

3. Run the readiness gate.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/check_prerelease_readiness.py \
  --bundle-dir out/prerelease_artifact_bundle/v1.26.0-rc1 \
  --output-dir out/prerelease_readiness/v1.26.0-rc1
```

4. Attach or link release assets.

- `out/prerelease_artifact_bundle/v1.26.0-rc1/prerelease_artifact_bundle.json`
- `out/prerelease_artifact_bundle/v1.26.0-rc1/prerelease_artifact_bundle.md`
- `out/prerelease_artifact_bundle/v1.26.0-rc1/SHA256SUMS`
- `out/prerelease_readiness/v1.26.0-rc1/prerelease_readiness.json`
- `out/prerelease_readiness/v1.26.0-rc1/prerelease_readiness.md`

5. Fill the release draft in `docs/releases/v1_26_prerelease_draft.md`.

## Required Public Links

Public materials must link:

- `docs/provenance.md`
- `docs/compliance.md`
- `docs/prerelease_artifact_bundle.md`
- `docs/prerelease_readiness.md`
- `docs/research_preview.md`
- `docs/rocm.md`
- `docs/CLAIMS.md`
- `docs/GETTING-STARTED.md`
- `docs/rocm_timing.md`
- `docs/RESEARCHER-GUIDE.md`
- `docs/v1_25_release_notes.md`
- `docs/v1_25_prerelease_checklist.md`

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
completed full MI300X validation on the CDNA3 `gfx942` target, or CDNA4
validation. Do not imply NVIDIA or AMD endorsement.

MI300X is the concrete CDNA3 `gfx942` hardware target, not a separate
architecture target. Full MI300X validation remains deferred until complete
real-hardware evidence exists. CDNA4 validation is unavailable because suitable
hardware is not currently accessible.

## Known Limitations To Include

- Trace JSONL remains canonical; sidecars are diagnostic-only unless explicitly
  narrower.
- Bounded dataset slices are provisional research-preview evidence, not paper
  parity.
- Docker/container ROCm user-space evidence is not native-host validation.
- No hosted leaderboard or remote submission service is included.
- No hardened multi-tenant sandbox is included.
