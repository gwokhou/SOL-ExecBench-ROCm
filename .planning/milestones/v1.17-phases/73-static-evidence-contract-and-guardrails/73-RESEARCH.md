# Phase 73: Static Evidence Contract And Guardrails - Research

**Researched:** 2026-05-25  
**Domain:** Python/Pydantic diagnostic sidecar contract, evaluator capability metadata, and canonical-output guardrails  
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### Sidecar Contract Shape
- Put the sidecar schema and helpers in
  `src/sol_execbench/core/bench/static_kernel_evidence.py` so diagnostic static
  evidence follows the existing profiler sidecar boundary and stays separate
  from canonical trace/data schemas.
- Represent authority boundaries with explicit schema booleans:
  `diagnostic_only=true`, `correctness_authority=false`,
  `performance_authority=false`, `timing_authority=false`,
  `score_authority=false`, `paper_parity_authority=false`, and
  `leaderboard_authority=false`.
- Use strict Pydantic models with stable enums and round-trip tests so
  downstream consumers cannot silently accept ambiguous sidecar payloads.
- Serialize skipped, unavailable, unsupported, partial, and failed outcomes as
  full valid sidecars with status and reason codes, even when no artifact is
  collected.

### Status And Reason Semantics
- Lock the top-level status vocabulary to `collected`, `partial`,
  `unavailable`, `unsupported`, `failed`, and `skipped`; subtype details belong
  in reason codes.
- Model reason codes as stable enum-like string values grouped by category,
  with future values added deliberately through tests.
- Keep artifact, tool-run, kernel, warning, and source-reference fields present
  as stable shapes, using empty lists when the section has no entries.
- Define conservative optional classification fields now, including metadata
  presence, disassembly presence, detected architectures, and symbol count; the
  extractor phases populate them later.

### Integration Guardrails
- Expose `static_kernel_evidence.v1` through existing evaluator contract
  optional capability metadata without changing the required evaluator contract
  version.
- Add negative guardrails proving canonical trace JSONL dumps, default CLI
  behavior, scoring artifacts, and sidecar generation remain isolated from the
  new static-evidence contract.
- Do not add CLI flags in Phase 73. Public `--static-evidence none|auto`
  belongs to Phase 76 after contract, discovery, and extractor plumbing exist.
- Explicitly defer artifact discovery, extractor subprocesses, report
  rendering, live ROCm validation, RGA-rich resource parsing, and Triton cache
  capture.

### the agent's Discretion
No open implementation choices require user input. Use nearby Pydantic,
sidecar, contract, and test patterns to choose exact class names and helper
function names.

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Artifact discovery from HIP/C++ staging/build trees.
- Durable evidence directory and artifact manifest population.
- Routed `llvm-objdump`, `readelf`, RGA, or `roc-objdump` subprocess execution.
- Public CLI flags and human-facing report rendering.
- Live ROCm validation artifacts.
- RGA-rich resource parsing and Triton ROCm cache capture.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SKE-CONTRACT-01 | Maintainer can serialize and parse a strict `sol_execbench.static_kernel_evidence.v1` sidecar schema with stable status, reason-code, artifact, tool-run, and classification fields. | Use a new strict Pydantic module under `core/bench`; Pydantic supports `extra='forbid'`, `strict=True`, frozen models, and JSON-compatible `model_dump(mode='json')`. [VERIFIED: repo grep] [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/] [CITED: https://pydantic.dev/docs/validation/latest/concepts/serialization/] |
| SKE-CONTRACT-02 | Maintainer can represent static evidence authority boundaries in the sidecar, with diagnostic-only semantics and explicit false authority for correctness, performance, timing, score, paper parity, and leaderboard claims. | Mirror `ToolchainRoutingReport` authority booleans and expand them to the locked static-evidence set. [VERIFIED: repo grep] |
| SKE-CONTRACT-03 | Maintainer can record aggregate and per-artifact static evidence states for collected, partial, unavailable, unsupported, failed, and skipped outcomes. | Model status as a `str, Enum`, require full sidecars for every outcome, and keep list sections present with empty defaults. [VERIFIED: repo grep] [CITED: https://pydantic.dev/docs/validation/latest/concepts/fields/] |
| SKE-CONTRACT-04 | Consumer can discover static evidence support through evaluator contract optional capability metadata without changing the required evaluator contract version. | Add only `static_kernel_evidence.v1` to `EvaluatorContract.capabilities`; keep `SOL_EXECBENCH_CONTRACT_VERSION == "1.0"`. [VERIFIED: repo grep] |
| SKE-CONTRACT-05 | Maintainer can verify that static evidence models and sidecar helpers do not mutate canonical trace JSONL, correctness, timing, scoring, or default benchmark behavior. | Extend existing public contract guardrail style: exact trace keys, primary CLI help exclusion, scoring artifact exclusion, and pure model serialization tests. [VERIFIED: repo grep] |
</phase_requirements>

## Summary

Phase 73 should add only the data contract and guardrails for `sol_execbench.static_kernel_evidence.v1`; no runtime discovery, tool execution, CLI flag, report writer, or live ROCm validation belongs here. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] The implementation should be a new Pydantic schema module in `src/sol_execbench/core/bench/static_kernel_evidence.py`, plus a small evaluator-contract capability update and focused tests. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] [VERIFIED: src/sol_execbench/core/data/contract.py]

The closest analog is `src/sol_execbench/core/bench/rocm_profiler.py`: it builds diagnostic sidecar payloads with schema versions, explicit non-authority booleans, status metadata, artifacts, command provenance, and nonfatal unavailable/failed outcomes. [VERIFIED: repo grep] The second analog is `src/sol_execbench/core/toolchain.py`, which already uses strict-ish Pydantic models, stable enum vocabularies, authority booleans, `Field(default_factory=list)` for stable list shapes, and `model_dump(mode="json")`. [VERIFIED: repo grep] Pydantic documentation confirms that `extra='forbid'` rejects unexpected fields, `strict=True` disables broad coercion, `frozen=True` prevents assignment, and `model_dump(mode='json')` emits JSON-compatible types. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/] [CITED: https://pydantic.dev/docs/validation/latest/concepts/serialization/]

**Primary recommendation:** Implement a strict, frozen Pydantic sidecar contract and pure helper constructors for `collected`, `partial`, `unavailable`, `unsupported`, `failed`, and `skipped`, then add guardrail tests proving no new static-evidence fields appear in canonical `Trace`, primary CLI help, scoring artifacts, or default benchmark behavior. [VERIFIED: repo grep]

## Project Constraints (from AGENTS.md)

- Python package code lives under `src/sol_execbench/`; tests live under `tests/sol_execbench/` for package behavior. [VERIFIED: AGENTS.md]
- Use Python 3.12+ and Ruff style; keep changes consistent with nearby modules and avoid broad refactors. [VERIFIED: AGENTS.md]
- Use `snake_case` for functions/variables/modules and `PascalCase` for classes and Pydantic models. [VERIFIED: AGENTS.md]
- Pytest is the test framework; prefer small unit tests for schema and driver logic. [VERIFIED: AGENTS.md]
- Environment-sensitive ROCm/GPU tests use existing markers, but Phase 73 should remain CPU-safe because no live ROCm validation is in scope. [VERIFIED: AGENTS.md] [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, or downloaded datasets. [VERIFIED: AGENTS.md]
- GSD workflow says repo edits should happen through a GSD workflow unless explicitly bypassed; this turn is a GSD research artifact write requested by the user. [VERIFIED: AGENTS.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Static evidence sidecar schema | Python package / data-contract layer | Bench diagnostics layer | The sidecar is a GPU-free serialization contract, not an evaluator execution path. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] |
| Status/reason vocabulary | Python package / data-contract layer | Future extractor phases | Phase 73 owns vocabulary stability; later phases only populate values. [VERIFIED: .planning/REQUIREMENTS.md] |
| Authority boundary booleans | Python package / data-contract layer | Docs in later phase | Contract payload must encode diagnostic-only semantics before reports/docs consume it. [VERIFIED: .planning/REQUIREMENTS.md] |
| Evaluator capability token | Evaluator contract metadata | CLI `contract --json` output | Capability discovery already flows through `build_evaluator_contract()` and `sol-execbench contract --json`. [VERIFIED: src/sol_execbench/core/data/contract.py] [VERIFIED: tests/sol_execbench/test_contract.py] |
| No-mutation guardrails | Tests | Canonical trace/scoring/CLI modules | Existing guardrails assert exact canonical keys and absence of derived option/key spaces. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] [VERIFIED: tests/sol_execbench/test_trace_reporting_and_score_guardrails.py] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.12.13 | Runtime for package and tests | Project requires Python 3.12+ and local environment reports 3.12.13. [VERIFIED: AGENTS.md] [VERIFIED: `python3 --version`] |
| Pydantic | 2.12.5 | Strict sidecar and evaluator-contract models | Existing data contracts use Pydantic v2 models; local environment and lockfile resolve 2.12.5. [VERIFIED: pyproject.toml] [VERIFIED: `uv run python -c importlib.metadata.version`] |
| Pytest | 9.0.2 | CPU-safe unit and guardrail tests | Existing test suite uses Pytest; local environment resolves 9.0.2. [VERIFIED: AGENTS.md] [VERIFIED: `uv run pytest --version`] |
| Ruff | 0.15.14 | Lint/format compatibility | Repo style is Ruff-enforced; local environment resolves 0.15.14. [VERIFIED: AGENTS.md] [VERIFIED: `uv run ruff --version`] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Click | 8.3.1 | Existing CLI contract/help guardrails | Use only in tests proving no Phase 73 CLI flag appears; do not add static-evidence options. [VERIFIED: pyproject.toml] [VERIFIED: `uv run python -c importlib.metadata.version`] |
| Rich | 14.3.3 | Existing CLI output dependency | No Phase 73 code should need new Rich output, but default CLI behavior tests may import the CLI. [VERIFIED: pyproject.toml] [VERIFIED: `uv run python -c importlib.metadata.version`] |
| Ty | 0.0.39 | Existing type-check command | Optional validation command if planner wants a type-check gate for changed files. [VERIFIED: pyproject.toml] [VERIFIED: `uv run ty --version`] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New Pydantic models in `core/bench/static_kernel_evidence.py` | Dataclasses with `to_dict()` like `rocm_profiler.py` | Dataclasses match older sidecar style, but locked decision requires strict Pydantic models and strict parsing. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] |
| Add sidecar fields to canonical `Trace` | Extend `Trace` model | Out of scope and violates canonical trace JSONL guardrails. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |
| CLI flag now | `--static-evidence none\|auto` in primary CLI | Deferred to Phase 76; Phase 73 must prove primary CLI help is unchanged. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] |

**Installation:** No new packages should be installed in Phase 73. [VERIFIED: .planning/research/SUMMARY.md] Existing dependencies are already in `pyproject.toml` and `uv.lock`. [VERIFIED: pyproject.toml] [VERIFIED: uv.lock]

## Package Legitimacy Audit

No external packages are added by this phase, so the Package Legitimacy Gate is not triggered. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] The phase uses already-declared project dependencies only. [VERIFIED: pyproject.toml]

## Architecture Patterns

### System Architecture Diagram

```text
Maintainer imports/constructs static sidecar models
    |
    v
StaticKernelEvidenceSidecar (strict Pydantic)
    |-- authority booleans fixed to diagnostic-only boundary
    |-- aggregate status + reason codes
    |-- stable empty-list sections for artifacts/tool_runs/kernels/warnings/source_refs
    |-- conservative optional classification metadata
    |
    +--> model_dump(mode="json") / model_validate(...)
    |
    +--> evaluator contract capability token: static_kernel_evidence.v1
    |
    v
Guardrail tests
    |-- canonical Trace keys unchanged
    |-- contract_version remains 1.0
    |-- primary CLI help has no --static-evidence
    |-- scoring/correctness/timing artifacts do not consume static sidecar data
```

### Recommended Project Structure

```text
src/sol_execbench/core/
├── bench/
│   ├── rocm_profiler.py              # Existing diagnostic sidecar analog. [VERIFIED: repo grep]
│   └── static_kernel_evidence.py     # New Phase 73 contract models/helpers. [VERIFIED: CONTEXT locked decision]
└── data/
    └── contract.py                   # Add optional capability token only. [VERIFIED: repo grep]

tests/sol_execbench/
├── test_static_kernel_evidence.py    # New schema, status, helper, and authority tests. [VERIFIED: planner recommendation]
├── test_contract.py                  # Add optional capability/version guardrail. [VERIFIED: repo grep]
└── test_public_contract_guardrails.py # Add negative canonical/CLI/static option guardrails. [VERIFIED: repo grep]
```

### Pattern 1: Strict Frozen Pydantic Sidecar Models

**What:** Define all sidecar payload shapes as Pydantic models with `ConfigDict(extra="forbid", frozen=True, strict=True, use_attribute_docstrings=True)`. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/]  
**When to use:** Use for the top-level sidecar and nested shapes because the phase requires strict parsing and round-trip tests. [VERIFIED: .planning/REQUIREMENTS.md]  
**Example:**

```python
# Source: Pydantic config docs + local BaseModelWithDocstrings pattern.
from enum import Enum

from pydantic import ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings

STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION = "sol_execbench.static_kernel_evidence.v1"


class StaticEvidenceStatus(str, Enum):
    COLLECTED = "collected"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    SKIPPED = "skipped"


class StaticEvidenceArtifact(BaseModelWithDocstrings):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        use_attribute_docstrings=True,
    )

    artifact_id: str
    kind: str
    status: StaticEvidenceStatus
    reason_code: str | None = None
    source_path: str | None = None
    persisted_path: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)


class StaticKernelEvidenceSidecar(BaseModelWithDocstrings):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        use_attribute_docstrings=True,
    )

    schema_version: str = STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION
    status: StaticEvidenceStatus
    reason_code: str | None = None
    diagnostic_only: bool = True
    correctness_authority: bool = False
    performance_authority: bool = False
    timing_authority: bool = False
    score_authority: bool = False
    paper_parity_authority: bool = False
    leaderboard_authority: bool = False
    artifacts: list[StaticEvidenceArtifact] = Field(default_factory=list)
    tool_runs: list[object] = Field(default_factory=list)
    kernels: list[object] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
```

### Pattern 2: Full Valid Sidecars for Non-Collected Outcomes

**What:** Provide helper constructors such as `build_static_evidence_sidecar(...)`, `build_static_evidence_skipped(...)`, `build_static_evidence_unavailable(...)`, `build_static_evidence_unsupported(...)`, and `build_static_evidence_failed(...)` that always return a valid top-level sidecar with empty stable sections. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]  
**When to use:** Use whenever future phases cannot collect artifacts or have intentionally skipped static evidence. [VERIFIED: .planning/REQUIREMENTS.md]  
**Local analog:** `collect_rocprofv3_profile()` returns `unavailable` and `failed` result objects with JSON payloads instead of raising benchmark failures. [VERIFIED: src/sol_execbench/core/bench/rocm_profiler.py]  

### Pattern 3: Capability Metadata Without Contract Bump

**What:** Add `static_kernel_evidence.v1` to `EvaluatorContract.capabilities` only; do not change `SOL_EXECBENCH_CONTRACT_VERSION`. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]  
**When to use:** This is the only Phase 73 integration point into existing public metadata. [VERIFIED: .planning/REQUIREMENTS.md]  
**Local analog:** `runtime.evidence.v1`, `profiling.evidence.v1`, and `toolchain.routing.v1` are optional capabilities while `contract_version` remains `"1.0"`. [VERIFIED: src/sol_execbench/core/data/contract.py] [VERIFIED: tests/sol_execbench/test_contract.py]

### Anti-Patterns to Avoid

- **Adding static evidence to `Trace`:** Canonical trace JSONL top-level keys are exact and guarded. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]
- **Adding `--static-evidence` now:** Phase 76 owns the public flag after contract/discovery/extractor plumbing exists. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]
- **Boolean-only evidence status:** Requirement SKE-CONTRACT-03 needs aggregate and per-artifact states across six statuses, not just `available: true/false`. [VERIFIED: .planning/REQUIREMENTS.md]
- **Parsing or executing tools in the contract phase:** `llvm-objdump`, `readelf`, RGA, and `roc-objdump` execution is deferred. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]
- **Permissive extra fields:** Pydantic ignores extra fields by default, so strict sidecar models must explicitly forbid extras. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON contract validation | Manual dict key validation | Pydantic v2 models with enums, `extra="forbid"`, and `strict=True` | Pydantic already provides typed validation, error reporting, default factories, and JSON-compatible serialization. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/] [CITED: https://pydantic.dev/docs/validation/latest/concepts/serialization/] |
| Enum/status serialization | Free-form strings scattered through helpers | `str, Enum` classes for status and reason code vocabularies | Existing project uses enum-backed status vocabularies for trace and toolchain routing. [VERIFIED: src/sol_execbench/core/data/trace.py] [VERIFIED: src/sol_execbench/core/toolchain.py] |
| Canonical-output isolation checks | Manual review only | Existing Pytest guardrail style in `test_public_contract_guardrails.py` and `test_trace_reporting_and_score_guardrails.py` | Existing tests already encode exact-key and non-mutation checks. [VERIFIED: repo grep] |
| Tool discovery/extraction | Contract-phase subprocess stubs | Defer to Phases 74 and 75 | User decisions explicitly exclude discovery and extractor subprocesses. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] |

**Key insight:** Phase 73 is a contract-stability phase; the planner should spend effort on strict model boundaries and negative guardrails, not on ROCm artifact mechanics. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]

## Common Pitfalls

### Pitfall 1: Pydantic Extras Silently Accepted
**What goes wrong:** Downstream payloads with misspelled or speculative fields validate because Pydantic defaults to ignoring extra fields. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/]  
**Why it happens:** Existing `BaseModelWithDocstrings` does not set `extra="forbid"` globally. [VERIFIED: src/sol_execbench/core/data/base_model.py]  
**How to avoid:** Set `ConfigDict(extra="forbid", frozen=True, strict=True, use_attribute_docstrings=True)` on every new static-evidence model. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/]  
**Warning signs:** Tests validate unknown fields or accept `"collected "` / boolean-like strings for enums. [VERIFIED: planner recommendation]

### Pitfall 2: Authority Flags Are Incomplete
**What goes wrong:** The sidecar says diagnostic-only but omits a specific false authority flag, allowing readers to infer correctness, timing, scoring, paper-parity, or leaderboard meaning. [VERIFIED: .planning/REQUIREMENTS.md]  
**Why it happens:** Existing routing report has only some authority booleans, while Phase 73 requires a broader set. [VERIFIED: src/sol_execbench/core/toolchain.py]  
**How to avoid:** Assert all seven locked booleans in schema tests and constructor tests. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]  
**Warning signs:** Tests only assert `diagnostic_only` and `score_authority`. [VERIFIED: tests/sol_execbench/test_rocm_profiler.py]

### Pitfall 3: Capability Token Accidentally Bumps Required Contract
**What goes wrong:** Consumers treating `contract_version` as a compatibility gate see an unnecessary version bump. [VERIFIED: tests/sol_execbench/test_contract.py]  
**Why it happens:** Capability addition and required contract version are edited together. [VERIFIED: src/sol_execbench/core/data/contract.py]  
**How to avoid:** Add `static_kernel_evidence.v1` to optional capabilities and strengthen the existing version test. [VERIFIED: tests/sol_execbench/test_contract.py]  
**Warning signs:** `SOL_EXECBENCH_CONTRACT_VERSION` changes from `"1.0"`. [VERIFIED: src/sol_execbench/core/data/contract.py]

### Pitfall 4: Contract Helpers Write Files or Scan Artifacts
**What goes wrong:** Phase 73 accidentally starts discovery, sidecar writing, or subprocess work that belongs to later phases. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]  
**Why it happens:** The profiler analog includes collection and file output helpers, but static evidence is not ready for runtime integration yet. [VERIFIED: src/sol_execbench/core/bench/rocm_profiler.py]  
**How to avoid:** Keep Phase 73 helpers pure: build/validate/serialize models only. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]  
**Warning signs:** New code imports `subprocess`, `shutil.which`, scans `Path.rglob`, or edits CLI options. [VERIFIED: planner recommendation]

## Code Examples

### Round-Trip Schema Guardrail

```python
# Source: local test_contract.py style + Pydantic model_dump(mode="json") docs.
from pydantic import ValidationError

from sol_execbench.core.bench.static_kernel_evidence import (
    STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION,
    StaticEvidenceStatus,
    StaticKernelEvidenceSidecar,
    build_static_evidence_skipped,
)


def test_static_evidence_skipped_round_trips_with_authority_flags():
    sidecar = build_static_evidence_skipped(reason_code="static_evidence_not_requested")
    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION
    assert payload["status"] == StaticEvidenceStatus.SKIPPED.value
    assert payload["diagnostic_only"] is True
    assert payload["correctness_authority"] is False
    assert payload["performance_authority"] is False
    assert payload["timing_authority"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False
    assert payload["artifacts"] == []
    assert StaticKernelEvidenceSidecar.model_validate(payload) == sidecar


def test_static_evidence_rejects_unknown_fields():
    payload = build_static_evidence_skipped(
        reason_code="static_evidence_not_requested"
    ).model_dump(mode="json")
    payload["score_input"] = True

    with pytest.raises(ValidationError):
        StaticKernelEvidenceSidecar.model_validate(payload)
```

### Evaluator Contract Capability Guardrail

```python
# Source: tests/sol_execbench/test_contract.py
def test_evaluator_contract_advertises_static_evidence_without_bump():
    payload = build_evaluator_contract().model_dump(mode="json")

    assert "static_kernel_evidence.v1" in payload["capabilities"]
    assert payload["contract_version"] == "1.0"
```

### Primary CLI Non-Exposure Guardrail

```python
# Source: tests/sol_execbench/test_public_contract_guardrails.py
def test_primary_cli_does_not_expose_static_evidence_before_phase_76():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "--static-evidence" not in result.output
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Runtime/profiler evidence as ad hoc diagnostics | Explicit diagnostic sidecars with schema versions and non-authority flags | Existing v1.14/v1.16 codebase state | Static evidence should follow sidecar-first boundaries. [VERIFIED: .planning/research/SUMMARY.md] [VERIFIED: src/sol_execbench/core/bench/rocm_profiler.py] |
| Tool lookup embedded in feature code | Toolchain routing report models availability and provenance | Existing v1.16 codebase state | Phase 73 should only define fields that later routed extractors can populate. [VERIFIED: .planning/research/SUMMARY.md] [VERIFIED: src/sol_execbench/core/toolchain.py] |
| Canonical trace extension for derived evidence | Derived/diagnostic outputs remain outside `Trace` JSONL | Existing guardrail tests | Static evidence must be sidecar-only. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |

**Deprecated/outdated:**
- Treating static evidence as score/correctness/timing authority is explicitly out of scope. [VERIFIED: .planning/REQUIREMENTS.md]
- Direct static tool subprocess execution in Phase 73 is out of scope. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|

**If this table is empty:** All claims in this research were verified or cited; no user confirmation is needed before planning.

## Open Questions (RESOLVED)

1. **RESOLVED: Should Phase 73 export new models from `sol_execbench.core.bench.__init__`?**
   - What we know: Context says models should be importable from `sol_execbench.core.bench` or directly from the new module. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]
   - What's unclear: Existing `core/bench` package export policy was not inspected as a mandatory file. [VERIFIED: files_to_read scope]
   - Decision: Phase 73 should keep direct-module imports as the stable public path unless `src/sol_execbench/core/bench/__init__.py` already exports diagnostic bench helpers. If it exports nothing or does not exist, do not create a broad package export surface in this phase. [VERIFIED: planner resolution]

2. **RESOLVED: Exact reason-code vocabulary names**
   - What we know: Reason codes must be stable enum-like strings grouped by category. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]
   - What's unclear: The locked context does not prescribe exact reason-code strings. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]
   - Decision: Use this exact starter vocabulary and test the complete enum set: `static_evidence_not_requested`, `static_evidence_collected`, `partial_artifact_metadata`, `partial_disassembly_only`, `partial_metadata_only`, `artifact_unavailable`, `toolchain_unavailable`, `unsupported_solution_type`, `unsupported_architecture`, `unsupported_artifact_type`, `extractor_failed`, `extractor_timeout`, and `parser_failed`. Values are category-prefixed by outcome family and future additions require deliberate test updates. [VERIFIED: planner resolution]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Runtime and tests | yes | 3.12.13 | None needed. [VERIFIED: `python3 --version`] |
| uv | Running project commands | yes | 0.11.15 | Use system Python only for inspection; planner should use uv for tests. [VERIFIED: `uv --version`] |
| Pydantic | New sidecar models | yes | 2.12.5 | None needed; already installed. [VERIFIED: `uv run python -c importlib.metadata.version`] |
| Pytest | Validation suite | yes | 9.0.2 | None needed. [VERIFIED: `uv run pytest --version`] |
| Ruff | Lint validation | yes | 0.15.14 | None needed. [VERIFIED: `uv run ruff --version`] |
| Ty | Optional type validation | yes | 0.0.39 | Ruff+pytest are sufficient if Ty is not in final plan. [VERIFIED: `uv run ty --version`] |
| ROCm GPU/tools | Phase 73 implementation | not required | n/a | Phase 73 is CPU-safe contract work. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] |

**Missing dependencies with no fallback:** None for Phase 73. [VERIFIED: environment audit]  
**Missing dependencies with fallback:** None for Phase 73. [VERIFIED: environment audit]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Pytest 9.0.2. [VERIFIED: `uv run pytest --version`] |
| Config file | `pyproject.toml` has `[tool.pytest.ini_options]`. [VERIFIED: rg pyproject.toml] |
| Quick run command | `uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py -q` [VERIFIED: planner recommendation] |
| Full suite command | `uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py tests/sol_execbench/test_public_contract_guardrails.py -q` [VERIFIED: planner recommendation] |
| Lint command | `uv run ruff check src/sol_execbench/core/bench/static_kernel_evidence.py src/sol_execbench/core/data/contract.py tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_contract.py tests/sol_execbench/test_public_contract_guardrails.py` [VERIFIED: planner recommendation] |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| SKE-CONTRACT-01 | Strict schema rejects unknown fields and round-trips `model_dump(mode="json")` through `model_validate`. | unit | `uv run pytest tests/sol_execbench/test_static_kernel_evidence.py::test_static_evidence_sidecar_round_trips -q` | no, Wave 0 create. [VERIFIED: repo grep] |
| SKE-CONTRACT-02 | Authority fields are `diagnostic_only=True` and all correctness/performance/timing/score/paper parity/leaderboard flags are false. | unit | `uv run pytest tests/sol_execbench/test_static_kernel_evidence.py::test_static_evidence_authority_boundaries_are_explicit -q` | no, Wave 0 create. [VERIFIED: repo grep] |
| SKE-CONTRACT-03 | Aggregate and artifact statuses cover `collected`, `partial`, `unavailable`, `unsupported`, `failed`, and `skipped`. | unit | `uv run pytest tests/sol_execbench/test_static_kernel_evidence.py::test_static_evidence_status_vocabulary_is_stable -q` | no, Wave 0 create. [VERIFIED: repo grep] |
| SKE-CONTRACT-04 | Evaluator contract advertises `static_kernel_evidence.v1` while `contract_version` remains `1.0`. | unit/CLI | `uv run pytest tests/sol_execbench/test_contract.py::test_evaluator_contract_advertises_static_evidence_without_bump -q` | existing file, add test. [VERIFIED: tests/sol_execbench/test_contract.py] |
| SKE-CONTRACT-05 | Static evidence does not alter `Trace` keys, primary CLI help, scoring/correctness/timing artifacts, or default behavior. | guardrail | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q` | existing files, add checks. [VERIFIED: repo grep] |

### Sampling Rate

- **Per task commit:** Run the focused quick command and Ruff on changed files. [VERIFIED: planner recommendation]
- **Per wave merge:** Run the full suite command above. [VERIFIED: planner recommendation]
- **Phase gate:** Full focused suite plus `uv run pytest tests/sol_execbench/test_toolchain_routing.py -q` to ensure the static contract does not disturb v1.16 routing semantics. [VERIFIED: tests/sol_execbench/test_toolchain_routing.py]

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_static_kernel_evidence.py` - new schema/authority/status/helper tests for SKE-CONTRACT-01 through SKE-CONTRACT-03. [VERIFIED: repo grep]
- [ ] Add `static_kernel_evidence.v1` capability assertion to `tests/sol_execbench/test_contract.py` for SKE-CONTRACT-04. [VERIFIED: tests/sol_execbench/test_contract.py]
- [ ] Add static-evidence absence checks to public guardrail tests for SKE-CONTRACT-05. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No user identity or authentication is touched. [VERIFIED: phase scope] |
| V3 Session Management | no | No sessions or web state are touched. [VERIFIED: phase scope] |
| V4 Access Control | no | No access-control boundary changes are touched. [VERIFIED: phase scope] |
| V5 Input Validation | yes | Strict Pydantic validation with forbidden extra fields and stable enums. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/] |
| V6 Cryptography | no | No hashing, signing, encryption, or secrets are in Phase 73 scope. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] |
| V8 Data Protection | yes | Guardrails prevent diagnostic sidecar data from mutating canonical trace/scoring outputs. [VERIFIED: .planning/REQUIREMENTS.md] |
| V12 File and Resources | no | File writing, artifact discovery, and durable evidence directories are deferred. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Overposting / unexpected schema fields | Tampering | `extra="forbid"` and unknown-field tests. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/] |
| Misclassification of diagnostic evidence as authority | Repudiation / Information integrity | Explicit false authority flags and guardrail tests. [VERIFIED: .planning/REQUIREMENTS.md] |
| Canonical-output contamination | Tampering | Exact-key tests for `Trace`, CLI help non-exposure, and scoring/correctness/timing isolation. [VERIFIED: tests/sol_execbench/test_public_contract_guardrails.py] |

## Sources

### Primary (HIGH confidence)

- `.planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md` - locked phase boundary, implementation decisions, and deferred scope. [VERIFIED: repo read]
- `.planning/REQUIREMENTS.md` - SKE-CONTRACT-01 through SKE-CONTRACT-05 and v1.17 exclusions. [VERIFIED: repo read]
- `.planning/STATE.md` - milestone status and accumulated decisions. [VERIFIED: repo read]
- `.planning/research/SUMMARY.md` - milestone architecture and phase ordering context. [VERIFIED: repo read]
- `src/sol_execbench/core/bench/rocm_profiler.py` - diagnostic sidecar analog and nonfatal unavailable/failed outcomes. [VERIFIED: repo read]
- `src/sol_execbench/core/data/contract.py` - evaluator contract capability metadata and version guardrail. [VERIFIED: repo read]
- `src/sol_execbench/core/toolchain.py` - enum/status/routing-report model patterns and authority fields. [VERIFIED: repo read]
- `tests/sol_execbench/test_contract.py` - evaluator-contract test style. [VERIFIED: repo read]
- `tests/sol_execbench/test_toolchain_routing.py` - routing report authority/status test style. [VERIFIED: repo read]
- `tests/sol_execbench/test_public_contract_guardrails.py` and `tests/sol_execbench/test_trace_reporting_and_score_guardrails.py` - canonical-output guardrail style. [VERIFIED: repo read]

### Secondary (MEDIUM/HIGH confidence)

- Pydantic configuration docs - `extra`, `frozen`, and `strict` model configuration semantics. [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/]
- Pydantic serialization docs - `model_dump(mode='json')` emits JSON-compatible types. [CITED: https://pydantic.dev/docs/validation/latest/concepts/serialization/]
- Pydantic fields docs - `Field(default_factory=...)` for stable list/dict defaults. [CITED: https://pydantic.dev/docs/validation/latest/concepts/fields/]

### Tertiary (LOW confidence)

- None. [VERIFIED: source audit]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all dependencies are existing repo dependencies and local versions were verified. [VERIFIED: pyproject.toml] [VERIFIED: environment audit]
- Architecture: HIGH - phase boundary and local analogs are explicit. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md] [VERIFIED: repo grep]
- Pitfalls: HIGH - derived from locked exclusions, Pydantic defaults, and existing guardrails. [VERIFIED: repo read] [CITED: https://pydantic.dev/docs/validation/latest/api/pydantic/config/]
- Exact helper names: MEDIUM - context delegates exact names to implementation discretion. [VERIFIED: .planning/phases/73-static-evidence-contract-and-guardrails/73-CONTEXT.md]

**Research date:** 2026-05-25  
**Valid until:** 2026-06-24 for local architecture; re-check Pydantic docs and lockfile if dependency versions change before implementation. [VERIFIED: planner recommendation]
