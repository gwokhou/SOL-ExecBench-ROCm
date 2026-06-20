---
generated_by: gsd-map-codebase
focus: concerns
mapped_at: 2026-06-16
---

# Concerns

## Runtime Isolation

- `src/sol_execbench/driver/templates/eval_driver.py` runs submitted solution
  code in a subprocess staging directory, but this is not a hardened sandbox.
- Reward-hack checks reduce obvious manipulation risks, yet untrusted
  submissions still require Docker, VM, or dedicated host isolation.
- The eval driver imports PyTorch and user code in the same subprocess, so
  global monkey-patching and side effects must remain actively guarded.

## Long CLI Module

- `src/sol_execbench/cli/main.py` owns command parsing, input loading,
  subprocess orchestration, sidecar handling, metadata subcommands, and dataset
  migration commands.
- This is operationally convenient but makes local changes high-blast-radius.
- Focused helper extraction should preserve Click behavior and public output
  contracts.

## Generated Driver Complexity

- `src/sol_execbench/driver/templates/eval_driver.py` is a generated script with
  many responsibilities: stdout redirection, problem loading, user import,
  reward-hack checks, input generation, correctness, timing, trace emission,
  and shutdown behavior.
- Small changes can affect subprocess semantics, strict JSONL output, or
  hardware timing behavior.
- Driver changes need integration coverage in `tests/sol_execbench/driver/` and
  E2E-style tests.

## Hardware Evidence Boundaries

- Many docs and reports distinguish RDNA4, CDNA3, MI300X, MI308X, and CDNA4
  evidence scopes.
- Incorrect wording can overstate validation authority even when code is
  technically correct.
- Claim-boundary tests and docs such as `docs/CLAIMS.md`, `docs/rocm.md`, and
  `docs/research_preview.md` should be updated together with evidence changes.

## Dataset Runner Size

- `scripts/run_dataset.py` is a large operator script that coordinates dataset
  runs, output layout, skip/reuse decisions, profiling, derived reports, and
  summaries.
- Some logic has been moved into `src/sol_execbench/core/dataset/`, but the
  script remains a major integration surface.
- Changes should prefer reusable package helpers when possible.

## Native Build Fragility

- Native ROCm paths depend on ROCm user-space, HIP compiler tooling, headers,
  architecture flags, PyTorch extension behavior, and device access.
- `src/sol_execbench/driver/problem_packager.py` auto-injects offload
  architecture flags when possible, but local `LOCAL` target probing can still
  depend on `rocm_agent_enumerator` or `rocminfo`.
- Header-dependent categories such as CK and rocWMMA need marker-gated tests.

## Performance Measurement Risk

- Timing uses HIP-backed PyTorch APIs and optional profiler evidence.
- Clock-lock state, profiler overhead, fallback timing, missing sidecars, and
  reference OOM states are explicitly modeled because performance claims are
  easy to overstate.
- `bounded` evidence categories describe what the current artifacts can support;
  deferred categories stay visible for work that needs new hardware, larger
  runs, or stronger authority before any external claim can be upgraded.
- Tests under `tests/sol_execbench/test_profiler_timing_coverage.py` and RDNA4
  profiler tests should stay close to timing changes.

## Repository Hygiene

- `__pycache__/` files are visible in the checkout listing; avoid treating them
  as source or including them in docs.
- Downloaded datasets, local traces, and generated benchmark output belong
  under ignored/output paths and should not be committed.
- Examples are excluded from Ruff, so example correctness relies on example
  tests and CLI-path tests rather than formatter enforcement.

## Deferred Or Blocked Areas

- CDNA4 validation and NVFP4/MXFP4 Quant ROCm adaptation remain deferred in the
  docs and diagnostics.
- MI300X validation must not be inferred from MI308X/`gfx942` infrastructure
  evidence.
- Full paper parity, upstream SOLAR parity, leaderboard authority, and broader
  AMD hardware validation remain explicit non-claims unless new evidence is
  recorded.
