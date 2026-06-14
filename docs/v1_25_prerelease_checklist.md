# v1.25 Prerelease Checklist

This checklist prepares a v1.25 engineering prerelease candidate. It documents
the maintainer flow from a clean tree to a tagged release candidate; it does
not publish a package or create a release by itself.

Version labels in this document are intentionally specific:

- `v1.25` refers to the engineering-prerelease milestone and release notes.
- `v1.26.0-rc1` appears in later bundle/readiness commands because the public
  prerelease packaging workflow was added after the v1.25 checklist.
- The Python package version in `pyproject.toml` is separate from both labels.

## 1. Confirm Tree And Version Context

```bash
git status --short
git log --oneline -5
```

Expected result: the tree is clean except for intentional release edits, and
the latest commits belong to the v1.25 engineering-prerelease milestone.

## 2. Run CPU-Safe Release Guardrails

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/sol_execbench/test_research_release_docs.py \
  tests/sol_execbench/test_public_contract_guardrails.py \
  tests/sol_execbench/test_release_candidate_validation.py \
  -q
```

These checks cover public release wording, claim boundaries, and the bounded
release-candidate validation wrapper without requiring live GPU hardware.

## 3. Generate Release-Candidate Validation Evidence

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/release_candidate_validation.py \
  --output-dir out/release_candidate_validation
```

On a ROCm-capable host, optionally add smoke evidence:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/release_candidate_validation.py \
  --output-dir out/release_candidate_validation \
  --include-rocm-smoke
```

On a Docker-capable ROCm setup, optionally add container smoke evidence:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/release_candidate_validation.py \
  --output-dir out/release_candidate_validation \
  --include-docker-smoke
```

With local benchmark assets, optionally add a bounded dataset slice:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/release_candidate_validation.py \
  --output-dir out/release_candidate_validation \
  --include-dataset-slice \
  --dataset-dir data/SOL-ExecBench/benchmark \
  --dataset-limit 5
```

For v1.26 and later public prerelease preparation, wrap the validation evidence
in a versioned artifact bundle:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/build_prerelease_artifact_bundle.py \
  --version v1.26.0-rc1 \
  --output-dir out/prerelease_artifact_bundle/v1.26.0-rc1
```

Review `prerelease_artifact_bundle.json`, `prerelease_artifact_bundle.md`, and
`SHA256SUMS` before publishing. The bundle maps artifacts and evidence surfaces
to `canonical`, `diagnostic-only`, `provisional`, `deferred`, or `unavailable`
authority classes.

Run the prerelease readiness gate before preparing public release materials:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/internal/release/check_prerelease_readiness.py \
  --bundle-dir out/prerelease_artifact_bundle/v1.26.0-rc1 \
  --output-dir out/prerelease_readiness/v1.26.0-rc1
```

## 4. Review Claim Boundaries

Review claim boundaries before tagging:

- `docs/v1_25_release_notes.md`
- `docs/CLAIMS.md`
- `docs/rocm.md`
- `docs/release_candidate_validation.md`
- `docs/prerelease_artifact_bundle.md`
- `docs/prerelease_readiness.md`
- `docs/GETTING-STARTED.md`

Confirm the release still says:

- Trace JSONL is canonical.
- Sidecars are diagnostic-only unless explicitly narrower.
- Bounded slices and support rows are provisional prerelease evidence.
- Full paper validation, upstream SOLAR parity, leaderboard readiness, and hard
  sandbox authority are deferred.
- MI300X and MI308X are sibling GPU products under the CDNA3 architecture
  family and share the `gfx942` code path; current CDNA3/gfx942 validation
  infrastructure evidence was recorded on MI308X, not MI300X, and full-suite
  MI300X validation is deferred without complete exact-hardware evidence.
- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.

## 5. Prepare The Candidate Commit

```bash
git status --short
git diff --check
```

Commit only intentional release edits and include the verification commands in
the commit or pull request notes.

## 6. Tag The Release Candidate

Use an annotated tag for a release candidate:

```bash
git tag -a v1.25.0-rc1 -m "v1.25.0-rc1"
```

The final release tag can use the project tag convention selected by the
maintainer, but do not tag until the validation evidence and claim boundaries
above have been reviewed.

## 7. Push Main And Tags

```bash
git push origin main --tags
```

After pushing, attach or link the generated release-candidate validation
artifacts in the release notes or pull request discussion. Do not present
optional ROCm, Docker, or dataset evidence as paper parity, hosted leaderboard
readiness, native-host validation, or unavailable hardware validation.
