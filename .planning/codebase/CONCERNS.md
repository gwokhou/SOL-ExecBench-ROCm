---
generated_at: 2026-06-02
last_mapped_commit: 8019adc6295a78d4636037889245abcb3f9a52bb
focus: concerns
---

# Concerns

## Status Map

- Narrowed: code-actionable concerns are now mostly concentrated in generated-driver complexity, evidence authority drift, and regression coverage.
- Externally blocked/deferred: full paper validation, MI300X validation on CDNA3, CDNA4 validation, native-host Matrix authority, hosted leaderboard readiness, and hard sandboxing require external evidence, hardware, or infrastructure.

## Release Claim Boundaries

- Full 235-problem paper validation is not complete and must not be claimed.
- Upstream NVlabs/SOLAR parity is not established and must not be claimed.
- Hosted leaderboard readiness and score authority are not established.
- Docker Matrix rows are container ROCm user-space evidence only; they are not native-host validation.
- MI300X is the concrete CDNA3 `gfx942` target, but full-suite MI300X validation on CDNA3 remains deferred without archived real-hardware evidence.
- CDNA4 validation is unavailable because suitable hardware is not currently accessible.

## Execution Isolation

- The evaluation subprocess is not a hardened sandbox.
- Reward-hack checks are meaningful guardrails but cannot provide multi-tenant security.
- Running untrusted submissions still requires a container, VM, or isolated ROCm host.
- Native compilation remains a sensitive boundary; compile flag validation in `src/sol_execbench/core/data/solution.py` should be kept strict when adding new native categories.

## Generated Driver Complexity

- `src/sol_execbench/driver/templates/eval_driver.py` remains large and behavior-dense.
- Recent helper extraction moved reusable runtime behavior to `src/sol_execbench/core/bench/eval_runtime.py`, but driver-level changes still need focused regression tests.
- stdout/stderr routing is fragile because strict Trace JSONL depends on redirecting non-JSON output before importing GPU libraries.

## Evidence Authority Drift

- The project has many diagnostic sidecars: profile, static evidence, Matrix, closure, scoring, consistency, stability, claim-upgrade, trust summary, and readiness reports.
- Each sidecar needs explicit authority-class wording so it is not mistaken for canonical trace output or benchmark score authority.
- `scripts/check_prerelease_readiness.py` now guards several forbidden claims, but new release docs and sidecars should update the gate when they introduce new claim surfaces.

## Test Coverage Gaps

- CI is intentionally CPU-safe and skips GPU e2e surfaces.
- Full driver e2e, native HIP/C++ builds, ROCm library examples, and real profiler behavior need local GPU or Docker validation.
- CDNA3 tests are marker-gated; absence of MI300X hardware means code/schema support can regress unless CPU-safe guardrails and later real-hardware validation are maintained.
- Timing and clock behavior can be environment-sensitive and should not be overinterpreted without clock policy evidence.

## ROCm Version And Hardware Matrix

- The dependency lock targets ROCm 7.1 PyTorch wheels, while Docker target rows include ROCm 7.0.2, 7.1.1, and 7.2.0.
- Mixed-version debugging exists, but mixed-version output cannot create clean validation, paper parity, or score authority.
- Native-host ROCm minor-version validation remains separate from container evidence.

## Legacy NVIDIA Residue

- Legacy CUDA/NVIDIA schema values are rejected, but residue examples and docs still exist for compatibility, migration, or provenance reasons.
- Files with NVIDIA attribution are controlled by `provenance.toml`; future cleanup should update both headers and the manifest.
- Legacy example directories such as `examples/cutlass/`, `examples/cudnn/`, `examples/cutile/`, and `examples/cute_dsl/` should not be mistaken for supported ROCm runtime paths.

## Suggested Milestone Candidates

- Engineering stable prerelease hardening: run and archive full ROCm/Docker smoke evidence, generated driver e2e, prerelease readiness, and release candidate validation from a clean tree.
- MI300X-on-CDNA3 validation milestone: run the adapted suite on real MI300X `gfx942`, archive environment, clock, trace, sidecar, and failure accounting evidence.
- Research benchmark authority milestone: complete paper denominator execution, upstream SOLAR comparison, score eligibility review, and reproducible artifact bundle.
- Execution-boundary hardening milestone: continue shrinking `eval_driver.py`, expand adversarial tests, and document residual sandbox limits.
- Provenance stewardship milestone: rerun upstream comparison when upstream moves, update `provenance.toml`, and keep source headers aligned without rewriting git history unless legal counsel requires it.
