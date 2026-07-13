---
phase: 116
phase_name: "Claim Boundary Guardrails"
created_at: "2026-06-01"
autonomous: true
requirements: [CLAIM-01, CLAIM-02, CLAIM-03]
---

# Phase 116 Context

## Goal

Keep v1.25 engineering-prerelease release wording bounded to the evidence this
repository actually has. A reader should not infer paper parity, upstream SOLAR
parity, leaderboard readiness, hard-sandbox authority, native-host validation
from Docker/container evidence, CDNA4 validation, or completed MI300X-on-CDNA3
full-suite validation.

## Inputs

- Phase 114 created `scripts/release_candidate_validation.py` and
  `docs/internal/release_candidate_validation.md` for bounded prerelease validation.
- Phase 115 created the public support matrix in `docs/user/rocm.md` and mirrored
  support boundaries in `docs/user/CLAIMS.md`.
- Existing docs already contain claim-boundary surfaces:
  - `docs/user/CLAIMS.md`
  - `docs/internal/release_candidate_validation.md`
  - `docs/internal/v1_19_evidence_guide.md`
  - `docs/internal/v1_20_evidence_quality_guide.md`
  - `docs/internal/v1_15_release_closure.md`
  - `docs/user/RESEARCHER-GUIDE.md`

## Locked Decisions

- v1.25 is an engineering prerelease, not paper-scale validation.
- Trace JSONL remains the canonical run artifact.
- Environment, profiling, static, Matrix, closure, consistency,
  claim-upgrade, trust-summary, and release-candidate summaries are sidecar or
  diagnostic evidence unless a narrower document says otherwise.
- MI300X is the concrete CDNA3 hardware target (`gfx942`); full-suite
  MI300X-on-CDNA3 validation is deferred without a complete real-hardware evidence
  chain.
- CDNA4 validation is unavailable because suitable hardware is not currently
  accessible.
- Docker/container ROCm user-space evidence is not native-host validation.
- Do not add a leaderboard, submission service, hard sandbox, new schema, or
  live hardware requirement in this phase.

## Current Patterns

- Documentation guardrails live in
  `tests/sol_execbench/test_research_release_docs.py` and
  `tests/sol_execbench/test_public_contract_guardrails.py`.
- Guardrails usually use direct substring checks against Markdown docs.
- Earlier phases prefer explicit negative wording over implied caveats.

## Risks

- Release wording can become scattered across docs, causing one page to imply a
  stronger claim than another page allows.
- "Engineering prerelease" can be misread as release authority if artifact
  classes are not named as canonical, diagnostic-only, provisional, or
  deferred.
- CDNA4 can be incorrectly described as merely untested rather than
  unavailable due to no accessible hardware.

## Deferred

- Running full 235-problem paper validation.
- Running MI300X-on-CDNA3 full-suite validation.
- Running CDNA4 validation.
- Implementing hosted leaderboard or hard multi-tenant sandbox operations.
- Changing trace, score, timing, Matrix, or validation schemas.
