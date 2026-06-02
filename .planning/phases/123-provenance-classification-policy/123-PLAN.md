# Phase 123: Provenance Classification Policy - Plan

## Goal

Create a policy and manifest that let maintainers classify active files by
provenance before any bulk SPDX/copyright cleanup.

## Tasks

1. Add a machine-readable `provenance.toml` manifest.
   - Record project attribution text.
   - Record source roots covered by the policy.
   - Record NVIDIA notice allowed files.
   - Record cleanup candidates that currently carry NVIDIA-only notices but do
     not exist at the same upstream path.

2. Add `docs/provenance.md`.
   - Define upstream retained, derivative modified, independent ROCm work, and
     generated/planning material.
   - Explain header policy for each class.
   - Explain paper citation versus source copyright.
   - Explain why ordinary commits, not history rewriting, are the planned
     correction path.

3. Add focused tests.
   - Parse the manifest with stdlib `tomllib`.
   - Verify all current NVIDIA SPDX header files are classified as allowed or
     cleanup candidates.
   - Verify allowed entries exist and keep a NVIDIA header today.
   - Verify cleanup candidates exist and keep a NVIDIA header today.

4. Verify.
   - Run the focused provenance policy test.
   - Run Ruff on touched Python test code.

## Non-Goals

- Do not change existing source headers in this phase.
- Do not change project license.
- Do not implement prerelease readiness gate changes yet.
- Do not perform a legal audit.

## Success Criteria

- PROV-01: Active files can be classified through documented classes and a
  reviewable manifest.
- PROV-02: The policy maps each class to header handling.
- PROV-03: The manifest identifies active files allowed to retain NVIDIA
  notices before bulk cleanup.
