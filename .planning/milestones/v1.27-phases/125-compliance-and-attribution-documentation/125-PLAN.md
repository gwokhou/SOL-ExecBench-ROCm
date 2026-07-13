# Phase 125: Compliance And Attribution Documentation - Plan

## Goal

Update public docs so the cleaned provenance policy is visible and release
materials avoid ownership or endorsement ambiguity.

## Tasks

1. Update `docs/user/compliance.md`.
   - Explain fork relationship, upstream notices, project-owned ROCm
     contributions, paper citation boundary, and non-endorsement wording.

2. Update README documentation links.
   - Add `docs/user/provenance.md`.
   - Ensure compliance/provenance are easy to find from the main entry point.

3. Update research and prerelease materials.
   - Add provenance/attribution sections.
   - Clarify paper citation versus file-level copyright.
   - Clarify no NVIDIA or AMD endorsement.

4. Verify.
   - Add or update focused documentation tests where useful.
   - Run focused docs tests and Ruff if touched tests change.

## Non-Goals

- Do not implement readiness gate enforcement.
- Do not change the release artifact schema.
- Do not change benchmark runtime behavior.
