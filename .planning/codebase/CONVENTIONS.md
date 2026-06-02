---
generated_at: 2026-06-02
last_mapped_commit: 8019adc6295a78d4636037889245abcb3f9a52bb
focus: quality
---

# Conventions

## Python Style

- Python source targets Python 3.12+.
- Ruff is the formatter/linter authority through `pyproject.toml`.
- Project modules, functions, variables, and files use `snake_case`.
- Classes, dataclasses, and Pydantic models use `PascalCase`.
- Enum classes use `PascalCase`; enum values are string values used in public schemas.
- Prefer local helper APIs over duplicating subprocess, JSON, checksum, and evidence logic.

## Source Headers

- Source attribution follows `provenance.toml` and `docs/provenance.md`.
- Upstream-retained and derivative-modified files keep applicable NVIDIA SPDX attribution.
- Modified derivative files also carry project SPDX attribution.
- Independent ROCm work carries project SPDX attribution only.
- `.planning/` and generated evidence are not treated like source files unless a source-style header is appropriate.

## Schema And Model Style

- Public schema models live in `src/sol_execbench/core/data/`.
- Pydantic validation is used for schema boundaries instead of ad hoc parsing.
- ROCm migration constraints are enforced in model validators where possible, for example legacy CUDA/NVIDIA language rejection in `src/sol_execbench/core/data/solution.py`.
- Public schema changes require docs and tests because examples, dataset scripts, and readiness gates consume these contracts.

## Error Handling

- CLI user-facing failures should use Click exceptions or clear Rich/console messages.
- Subprocess failures are converted to bounded diagnostic logs and no-trace sidecars when Trace JSONL cannot be parsed.
- Dataset runner failures persist bounded CLI logs through helpers in `src/sol_execbench/core/dataset/runner.py`.
- Diagnostic reports should use explicit statuses such as `deferred`, `unavailable`, `diagnostic-only`, `unsupported`, or `blocking`.

## Subprocess And Filesystem Boundaries

- Benchmark execution happens in a staging directory created by `ProblemPackager`.
- User solution source paths must be relative and must not contain parent traversal.
- Native compile flags must not introduce response files, external include/library path injection, or runtime linker control.
- The generated driver redirects non-JSON stdout to stderr before importing PyTorch/Triton so stdout remains strict JSONL.

## Evidence Semantics

- Trace JSONL is canonical benchmark output.
- Environment, profile, static-kernel, Matrix, closure, consistency, claim-upgrade, trust-summary, and release-candidate artifacts are sidecar evidence unless a specific document narrows their role.
- Readiness gates should encode forbidden claims rather than relying only on prose.
- Docker/container evidence must not be described as native-host validation.

## Documentation

- Public behavior changes should update relevant docs under `docs/` and README links.
- Claim language must stay aligned with `docs/CLAIMS.md`.
- MI300X should be described as the concrete CDNA3 `gfx942` target.
- CDNA4 validation should be described as unavailable until suitable hardware is accessible.
