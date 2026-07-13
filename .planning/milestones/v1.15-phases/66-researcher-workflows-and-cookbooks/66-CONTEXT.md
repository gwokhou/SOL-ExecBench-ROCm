# Phase 66: Researcher Workflows and Cookbooks - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Give GPU kernel researchers and deep developers a direct path from first run to
extending kernels, interpreting artifacts, and running agent or compiler
experiments.
</domain>

<decisions>
## Implementation Decisions

- Split narrative guidance (`RESEARCHER-GUIDE.md`) from command recipes
  (`COOKBOOK.md`).
- Center the guide around researcher roles instead of vendor-specific details.
- Keep commands grounded in existing samples and examples.
</decisions>

<code_context>
## Existing Code Insights

- README already covers basic user setup.
- `docs/internal/analysis.md`, `docs/user/rocm.md`, and `docs/internal/original_parity.md` contain
  detailed artifact semantics but not a role-based entry point.
</code_context>

<specifics>
## Specific Ideas

- Add role paths for kernel authors, compiler/backend researchers, agent
  researchers, and benchmark/reproducibility researchers.
- Add recipes for one kernel, HIP/C++ adaptation, curated slice, environment
  evidence, profiling evidence, and AMD-native derived evidence.
</specifics>

<deferred>
## Deferred Ideas

- Do not build an agent optimizer loop in this milestone.
</deferred>

