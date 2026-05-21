# Phase 05 Context: ROCm Test Suite and Hardware Validation

## Goal

Convert the test suite to ROCm semantics and record hardware validation status for
RDNA 4 and CDNA 3 without claiming hardware evidence that has not been collected.

## User Decision

The selected scope is local-verifiable migration plus hardware matrix evidence:

- Migrate pytest markers, skip reasons, examples, and focused test assertions from
  CUDA/NVIDIA assumptions to ROCm/AMD assumptions.
- Run the tests that are feasible in this local environment.
- Require explicit RDNA 4 and CDNA 3 validation evidence before marking those
  hardware targets as passed.
- Use `uv run --no-sync ...` for local verification to avoid triggering a large
  dependency re-sync/download during this phase.

## Current Findings

- `tests/conftest.py` still detects NVIDIA SM versions and exposes
  `requires_sm100` / `requires_cutile` Blackwell semantics.
- `pyproject.toml` marker descriptions still describe C++/CUDA and cuTile.
- `tests/examples/test_examples.py` has ROCm solution schemas, but some fallback
  examples still carry stale `cpp` / `requires_cutile` markers.
- `tests/sol_execbench/test_e2e.py` still groups native languages as CUDA-era
  names and has a CUDA/C++ phase comment.
- `src/sol_execbench/cli/main.py` has user-facing C++/CUDA compile help text.
- Reward-hack tests remain present; skip messages should describe ROCm GPU
  availability while continuing to use PyTorch's `torch.cuda` compatibility API
  on ROCm.

## Boundaries

- Do not rename example directories or legacy `solution_cuda.json` filenames in
  this phase; Phase 4 intentionally preserved path compatibility while changing
  solution contents.
- Do not mark RDNA 4 or CDNA 3 as passed unless the adapted suite actually runs
  on that class of hardware.
- NVIDIA copyright headers and explicit audit allowlists are not semantic test
  failures.

