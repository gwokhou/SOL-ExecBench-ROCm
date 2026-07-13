# Phase 131: Dataset License and Provenance Policy - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

This phase defines enforceable source, license, provenance, and redistribution
boundaries before any migrated dataset artifacts are generated or integrated.
It covers machine-readable policy, documentation, and CPU-safe guardrails only;
local migration tooling and dataset-runner integration remain later phases.

</domain>

<decisions>
## Implementation Decisions

### Conservative Redistribution Boundary
- NVIDIA SOL-ExecBench original dataset content and ROCm-migrated derivatives
  are not project-redistributable. They are local-only inputs that users obtain
  and migrate under their own applicable NVIDIA Evaluation Dataset License
  rights.
- FlashInfer Trace is tracked separately as Apache-2.0 content from
  `flashinfer-ai/flashinfer-trace`; Apache attribution and notice requirements
  must remain distinct from the NVIDIA dataset boundary.
- Generated local migration artifacts inherit their source dataset boundary and
  must record source dataset id, revision, checksums, and license-boundary
  metadata.
- Project-owned ROCm code and metadata remain publishable Apache-2.0 project
  material unless an individual file records retained upstream expression.

### Enforcement Surface
- Extend `provenance.toml` rather than adding a separate policy file, so
  source/header provenance and dataset redistribution policy remain in one
  machine-readable manifest.
- Add a CPU-safe checker that can inspect staged repository paths and release
  bundle directories.
- Wire release-bundle scanning into prerelease readiness so restricted dataset
  payloads fail before publication.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/REQUIREMENTS.md` — DATA-LIC-01 through DATA-LIC-04 requirements.
- `.planning/ROADMAP.md` — Phase 131 goal and success criteria.
- `provenance.toml` — existing machine-readable provenance manifest.
- `docs/user/provenance.md` — public provenance policy.
- `docs/user/compliance.md` — public compliance and known-gap policy.
- `scripts/check_prerelease_readiness.py` — existing release gate.

</canonical_refs>

<code_context>
## Existing Code Insights

- `provenance.toml`, `docs/user/provenance.md`, and
  `tests/sol_execbench/test_provenance_policy.py` already enforce file-level
  source/header provenance from earlier milestones.
- `scripts/check_prerelease_readiness.py` already blocks missing provenance docs
  and claim-boundary regressions for prerelease bundles.
- `.gitignore` excludes `/data/*`, but forced-adds and release-bundle copies
  still need explicit CPU-safe guardrails.

</code_context>

<specifics>
## Specific Ideas

- Include redistribution classes: publishable, local-only, generated-only,
  excluded, and release-bundle-blocked.
- Include source entries for `nvidia/SOL-ExecBench`,
  `flashinfer-ai/flashinfer-trace`, generated local migration artifacts, and
  project-owned ROCm code.
- Tests should simulate forbidden paths without adding real dataset payloads.

</specifics>

<deferred>
## Deferred Ideas

- Actual local migration commands and manifests: Phase 132.
- ROCm readiness classification and ready subsets: Phase 133.
- Dataset runner consumption of license-boundary metadata: Phase 135.

</deferred>
