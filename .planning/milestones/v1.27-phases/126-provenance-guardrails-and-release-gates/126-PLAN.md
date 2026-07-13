# Phase 126: Provenance Guardrails And Release Gates - Plan

## Goal

Add automated provenance checks to release readiness and ensure existing audits
protect both incorrect NVIDIA-only headers and missing required NVIDIA notices.

## Tasks

1. Extend `scripts/check_prerelease_readiness.py`.
   - Parse `provenance.toml`.
   - Verify `docs/user/provenance.md` exists and has required phrases.
   - Verify allowed files contain NVIDIA and project attribution.
   - Verify cleanup candidates contain project attribution and no NVIDIA
     attribution.
   - Verify current active NVIDIA headers match the allowlist.

2. Extend readiness tests.
   - Passing bundle should include provenance checks.
   - Missing or malformed provenance manifest should produce blocking findings.

3. Verify.
   - Run readiness tests.
   - Run provenance policy and residue audit tests.
   - Run Ruff on touched Python files.

## Non-Goals

- Do not change bundle schema.
- Do not alter benchmark execution.
- Do not perform full REUSE conformance.
