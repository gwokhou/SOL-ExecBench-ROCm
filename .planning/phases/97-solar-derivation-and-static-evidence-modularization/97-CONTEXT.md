---
phase: 97
status: active
created: 2026-06-01
---

# Phase 97 Context: SOLAR Derivation And Static Evidence Modularization

## Goal

Reduce coupling in SOLAR derivation and static kernel evidence by extracting
status, provenance-boundary, warning, and extractor aggregation helpers while
preserving public sidecar schemas and diagnostic-only authority boundaries.

## Scope

- `src/sol_execbench/core/scoring/solar_derivation.py`
- `src/sol_execbench/core/bench/static_kernel_evidence.py`
- New focused helper modules under scoring/bench packages
- Focused tests in existing SOLAR/static evidence suites

## Non-Goals

- No sidecar schema changes.
- No authority or claim upgrades.
- No new hardware/toolchain validation.
- No rewrite of parser or extractor execution.

## Requirements

- ANALYSIS-03: SOLAR derivation separates semantic provenance, bound/formula
  derivation, coverage/status classification, and report rendering while
  preserving public sidecar schemas.
- ANALYSIS-04: Static evidence separates artifact discovery, tool routing,
  bounded capture, parser behavior, and sidecar/report rendering behind focused
  helpers and fixtures.
- ANALYSIS-05: Refactors preserve existing AMD score, AMD SOL/SOLAR, static
  evidence, and claim-boundary behavior unless explicitly changed.
