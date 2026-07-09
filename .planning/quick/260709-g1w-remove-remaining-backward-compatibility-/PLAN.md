---
status: complete
date: 2026-07-09
---

# Remove remaining backward compatibility facades

Task: Remove remaining backward compatibility facades and legacy parser fallbacks.

Plan:
- Fix stale console entry points that still target removed compatibility modules.
- Rewrite imports from flat bench/scoring facade modules to canonical package modules.
- Delete flat bench/scoring facade modules after import rewrites.
- Remove legacy parser/fallback branches that preserve older sidecar or graph inputs.
- Run focused import, CLI, scoring, and lint checks.

Result:
- Removed remaining flat compatibility facades and rewired imports to canonical packages.
- Removed stale CLI entry points and restored `sol_execbench.cli:cli` through a lazy package export.
- Removed legacy AMD SOL fallback estimation and legacy SOLAR sidecar parsing paths.
- Removed unused legacy clock preset API and tightened clock-lock function signatures.
- Updated provenance and residue guardrails for the new package layout.
