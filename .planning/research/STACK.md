# Stack Research

**Domain:** SOL ExecBench ROCm engineering-practice adaptation
**Researched:** 2026-05-22
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.12+ | Existing package, CLI, schemas, eval driver | Required to preserve the current public `sol-execbench` contract and Pydantic models. |
| Pydantic | v2 | Public schema validation | Current definition/workload/solution/trace contracts are Pydantic-based and should not be replaced. |
| Click + Rich | Existing dependency versions | CLI and human-readable terminal output | Existing public CLI uses these; additive output should stay consistent. |
| PyTorch ROCm events | Existing PyTorch ROCm pin | Timing and reference execution | Current timing semantics rely on HIP-backed `torch.cuda.Event` and unique tensor allocation. |
| Pytest | Existing dev dependency | Unit and E2E validation | Existing test suite and markers already distinguish RDNA 4, CDNA 3, ROCm, and slow compiled tests. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Standard library `dataclasses` / `Enum` | Python 3.12 | Stage result and diagnostic data models | Prefer for internal helper surfaces that must not affect public schema. |
| Standard library `json` / `pathlib` | Python 3.12 | Evidence and report files | Sufficient for internal validation manifests and trace summaries. |
| Existing `gsd-sdk` workflow tooling | Installed locally | Planning state and commits | Use only for planning artifacts, not runtime benchmark behavior. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `uv run pytest` | Unit and E2E test execution | RDNA 4 validation should include focused unit tests plus existing E2E coverage. |
| `uv run ruff check .` | Lint validation | Keeps new helpers aligned with current project style. |
| `./scripts/run_docker.sh` | ROCm Docker validation | Validation workflow readiness should document and test command construction without requiring CDNA 3 hardware. |

## Stack Additions

No new runtime dependency is recommended for v1.4. `hip-execbench` uses
TypeScript, Commander, Zod, Pino, and Plotly-style reporting, but importing that
stack would create a second framework and pressure public contracts. The
portable lessons are architectural, not dependency-level:

- Add internal stage-result data structures using Python dataclasses.
- Add optional evidence/report writers using existing JSON and Markdown tooling.
- Add validation readiness helpers using existing ROCm detection patterns.
- Add unit tests around compatibility guardrails before touching benchmark code.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Internal Python helpers | Port `hip-execbench` TypeScript modules | Only if the project intentionally adds a separate Node package, which is out of scope. |
| Additive reports over trace JSONL | New primary agent JSON contract | Only after an approved public contract change. |
| Existing PyTorch event timing | Standalone HIP binary timing as primary path | Only for an opt-in helper; not as a replacement for the paper-style eval driver. |
| Internal validation manifests | Mutating trace schema with validation metadata | Only after trace schema versioning is introduced. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Zod schema replacement | Would fork public Pydantic contracts. | Keep Pydantic schemas; use internal dataclasses for non-public structures. |
| Mandatory HTML/Plotly report dependencies | Adds frontend dependencies for a benchmark CLI and changes operator expectations. | Markdown or JSON evidence generated with existing dependencies. |
| Replacing eval driver with standalone HIP binary execution | Would weaken reference correctness, timing, and reward-hack semantics. | Adapt pipeline-result concepts around the existing eval driver. |
| Declaring CDNA 3 validation complete from readiness tests | Readiness is not hardware validation. | Keep explicit claim-level guardrails and evidence requirements. |

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `torch==2.10.0+rocm7.1` | Existing timing and eval driver | Do not change timing APIs in v1.4 unless tests prove compatibility. |
| Pydantic v2 | Existing schema models | Internal helpers should avoid becoming public model fields. |
| Pytest markers | `requires_rdna4`, `requires_cdna3`, `cpp`, `requires_rocm` | New tests should reuse marker semantics instead of inventing new naming. |

## Sources

- `src/sol_execbench/core/data/*.py` — existing public schema contracts.
- `src/sol_execbench/driver/templates/eval_driver.py` — reference correctness and reward-hack harness.
- `src/sol_execbench/core/bench/timing.py` — current timing semantics.
- `~/PyCharmMiscProject/hip-playground/hip-execbench/src/pipeline/runner.ts` — stage-result pipeline pattern.
- `~/PyCharmMiscProject/hip-playground/hip-execbench/src/compiler/hipcc.ts` — compile-cache and command logging discipline.

---
*Stack research for: v1.4 engineering-practice adaptation*
*Researched: 2026-05-22*
