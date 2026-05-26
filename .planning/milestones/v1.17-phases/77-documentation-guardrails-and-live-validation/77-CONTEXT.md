# Phase 77: Documentation, Guardrails, And Live Validation - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 77 documents Static Kernel Evidence, claim boundaries, CPU-safe test
coverage, deferred scopes, and bounded live RDNA 4 validation status. The phase
must not overstate static evidence as correctness, performance, timing, score,
paper-parity, or leaderboard authority.

</domain>

<decisions>
## Implementation Decisions

### Live Validation
- Attempt a bounded live RDNA 4 validation only when required ROCm build tools,
  runtime, and device access are available.
- If the current environment cannot support live validation, write an explicit
  skipped/unavailable validation artifact instead of blocking or claiming a
  hardware pass.

### Documentation
- Add `docs/static_kernel_evidence.md` as the primary user-facing guide.
- Update `docs/CLAIMS.md` and `docs/RESEARCHER-GUIDE.md`.
- Add a v1.17 validation artifact under `docs/internal/`.

### Deferred And Unsupported Scope
- Explicitly mark CDNA 3, CDNA 4, Triton cache capture, RGA-rich resource
  parsing, and paper-scale static coverage as unsupported, partial, or deferred
  unless direct evidence exists.
- Add tests so docs cannot overclaim these boundaries.

</decisions>
