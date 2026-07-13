# Quick Task Context

Task: Reorganize SOL ExecBench source and tests into module-aligned folders with compatibility facades.

User-approved scope:

- Full repository source/test organization, not only one package area.
- Group flat source files by module responsibility.
- Make tests mirror source paths where practical.
- Preserve existing public import paths through thin facades during the move.

Design spec:

- `docs/internal/superpowers/specs/2026-07-08-module-aligned-layout-design.md`

Execution constraints:

- Follow repository GSD workflow before edits.
- Keep changes focused on file organization, imports, facades, and matching test moves.
- Do not move fixtures, samples, Docker tests, or example workflow tests solely to force a source-tree mirror.
- Verify with focused pytest runs, package tests, full tests when feasible, and Ruff.
