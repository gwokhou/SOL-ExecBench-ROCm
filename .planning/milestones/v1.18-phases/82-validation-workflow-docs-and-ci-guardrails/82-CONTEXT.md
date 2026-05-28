# Phase 82: Validation Workflow, Docs, And CI Guardrails - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning
**Source:** $gsd-autonomous smart discuss, accepted by user

<domain>
## Phase Boundary

Users can follow documented validation workflows and trust automated guardrails
that prevent ROCm matrix evidence from being overstated.

This phase covers DOCS-01 through DOCS-06:

- Documentation must explain that Docker Matrix Entries validate container ROCm
  user-space on recorded host driver/devices and do not prove native host ROCm
  validation.
- Documentation must explain Target/requested values versus observed
  host/container/Python/GPU evidence, including why Target identity is required.
- Documentation must state that illegal mixed-version Targets are blocked by
  default and may only continue under explicit debug override without clean
  validation claims.
- CPU-safe tests must cover status classification, reason-code classification,
  schema serialization, mixed-version blocking, claim flags, docs wording
  guardrails, Docker Target selection, default preservation, unknown Target
  rejection, and command construction.
- Live ROCm validation guidance must be marker-gated and record the current host
  ROCm 7.1.x environment as observed evidence without requiring host reinstall
  for ROCm 7.0.x or 7.2.x.

</domain>

<decisions>
## Implementation Decisions

### Documentation scope

- Update existing documentation rather than adding a large new documentation
  tree.
- Use `docs/CLAIMS.md` for Matrix evidence claim boundaries and
  `docs/TESTING.md` for validation commands and marker-gated live guidance.
- Keep Docker Matrix language precise: container ROCm user-space on recorded
  host driver/devices is not native host validation.

### Target versus observed evidence

- Explain that Target/requested values identify what was selected, while
  observed evidence records what probes found at runtime.
- Explain that Matrix interpretation requires both values because mismatches are
  the compatibility signal.

### Debug override language

- Document mixed-version debug override as probe/smoke-only.
- Do not describe debug override runs as clean validation, score, paper-parity,
  or leaderboard evidence.

### Guardrail tests

- Add docs wording tests for claim boundaries and live validation guidance.
- Reuse existing CPU-safe schema/status/reason/claim/script tests rather than
  duplicating implementation coverage.
- Do not make default CI require live Docker, ROCm hardware, or host reinstall.

### Live validation guidance

- Document marker-gated live ROCm checks and the required evidence sidecars.
- Guidance may record the current host ROCm 7.1.x environment as observed
  evidence.
- ROCm 7.0.x and 7.2.x host-native validation remains deferred unless run on a
  matching host; Docker user-space validation must not be overstated.

</decisions>

<code_context>
## Existing Code Insights

- `docs/CLAIMS.md` already contains general claim boundaries and a short Docker
  Matrix statement.
- `docs/TESTING.md` already documents pytest markers such as `requires_rocm`.
- `tests/conftest.py` already defines marker-gated ROCm behavior and hardware
  architecture markers.
- Phases 78-81 already provide CPU-safe tests for Matrix schema, claim flags,
  Docker Target selection/preflight, dependency policy, runtime evidence
  sidecars, and Docker wrapper command construction.

</code_context>

<specifics>
## Specific Ideas

- Expand `docs/CLAIMS.md` with a concise "ROCm Compatibility Matrix" section.
- Expand `docs/TESTING.md` with "ROCm Matrix Guardrails" commands and
  marker-gated live evidence guidance.
- Add `tests/sol_execbench/test_rocm_matrix_docs.py` to assert required wording
  and guardrail command references.
- Update requirements, roadmap, state, and verification artifacts after tests
  pass.

</specifics>

<deferred>
## Deferred Ideas

- Milestone audit/completion/cleanup should run after Phase 82 verification, not
  as an implementation task inside Phase 82.
- CDNA3/CDNA4 real hardware validation remains future work.
- Native host ROCm 7.0.x/7.2.x validation remains future work unless executed on
  matching hosts.

</deferred>
