# Project Research: Architecture

## Milestone

v1.27 Copyright Provenance Cleanup

## Question

How should the cleanup integrate with the existing ROCm port architecture and
release gates?

## Existing Integration Points

- Active NVIDIA/CUDA residue audit:
  `tests/sol_execbench/test_rocm_migration_residue_audit.py`
- Prerelease readiness gate:
  `scripts/check_prerelease_readiness.py`
- Public artifact bundle:
  `scripts/build_prerelease_artifact_bundle.py`
- Compliance documentation:
  `docs/compliance.md`
- Release and research preview docs:
  `docs/public_prerelease.md`, `docs/research_preview.md`,
  `docs/releases/v1_26_prerelease_draft.md`
- Third-party notices:
  `THIRD_PARTY_NOTICES.txt`

## Suggested Shape

1. Create a provenance policy document.
   It should define file classes, header policy, upstream attribution, paper
   citation policy, and explicit non-endorsement wording.

2. Add a lightweight classification artifact.
   The artifact can be a small Markdown table, a JSON/TOML manifest, or a
   Python module constant consumed by tests. It must be reviewable by humans
   and stable enough for CI.

3. Refactor the residue audit.
   The audit should no longer classify every NVIDIA SPDX line as acceptable.
   It should permit NVIDIA notices only in upstream-retained or derivative
   files.

4. Add release-readiness integration.
   The readiness check should fail if:
   - active independent files contain NVIDIA-only file copyright;
   - derivative files drop required upstream NVIDIA notices;
   - provenance docs are missing;
   - public docs imply NVIDIA or AMD endorsement.

5. Update docs and release materials.
   Public-facing docs should say this project is an Apache-2.0 ROCm port of
   SOL-ExecBench, preserves benchmark semantics where practical, and cites the
   paper for method context.

## Build Order

1. Audit/classification.
2. Header and notice cleanup.
3. Test/gate refactor.
4. Documentation and release material alignment.

This order avoids editing file headers before the project has a defensible
classification rule.
