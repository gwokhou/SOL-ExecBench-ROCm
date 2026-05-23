# Phase 41: Bound Model Contract And Hardware Artifacts - Research

**Researched:** 2026-05-23
**Domain:** Python package resource loading, strict JSON artifact validation, AMD hardware model contract, public schema guardrails
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

### Artifact Location And Packaging

- **D-01:** Default AMD hardware model JSON artifacts must be packaged with the
  Python package under `src/sol_execbench/data/amd_hardware_models/`, starting
  with an RDNA 4 `gfx1200` artifact.
- **D-02:** Packaged artifacts should be loaded through Python package resource
  APIs such as `importlib.resources`, so defaults remain available after
  package installation.
- **D-03:** Phase 41 should implement loader support for an arbitrary external
  hardware model JSON path, but should not add dataset-runner or CLI flags yet.
  Wiring external paths into `scripts/run_dataset.py` remains Phase 45 scope.

### Hardware Validation Status Semantics

- **D-04:** v2 hardware model artifacts must distinguish hardware/environment
  validation from model validation with two fields:
  `hardware_validation_status` and `model_validation_status`.
- **D-05:** RDNA 4 hardware validation context may be represented separately
  from model validation, but the Phase 41 bound model itself remains
  provisional until Phase 46 records bound-model validation evidence.
- **D-06:** v2 artifacts hard-replace the old single `validation_status` field.
  `validation_status` is not part of the v2 contract. Legacy v1 compatibility
  code may translate from the old field internally, but v2 JSON must use the two
  explicit status fields.

### Artifact Schema Strictness

- **D-07:** v2 hardware model and bound artifact loaders must reject unknown
  fields instead of silently accepting schema drift.
- **D-08:** Invalid packaged artifacts should fail explicitly during loading or
  tests; they must not fall back to hard-coded model constants.

### Fallback Model Policy

- **D-09:** Hard-coded peak compute, memory bandwidth, source, and validation
  metadata must be moved out of `default_amd_hardware_models()` and into
  packaged JSON artifacts.
- **D-10:** `default_amd_hardware_models()` may remain as a compatibility API,
  but it must load packaged JSON artifacts. If packaged JSON is missing or
  invalid, it should fail explicitly rather than returning embedded constants.

### Public-Contract Guardrail Depth

- **D-11:** Phase 41 must guard canonical `Trace` JSONL, primary
  `sol-execbench` CLI behavior, and public definition/workload/solution schemas
  against accidental bound-modeling changes.
- **D-12:** Phase 41 must also add documentation or grep-style guardrails that
  block premature B200, upstream SOLAR, leaderboard-equivalence, CDNA 3 /
  MI300X validation, and CDNA 4 validation claims in v1.9 docs or outputs.
- **D-13:** Full score-report warning integration remains Phase 45 scope unless
  a small compatibility shim is required to keep existing tests passing.

### the agent's Discretion

The agent may choose exact module names, dataclass/Pydantic split, and test file
organization as long as the decisions above are preserved, existing public
imports are handled deliberately, and canonical trace/CLI contracts do not
change.

### Deferred Ideas (OUT OF SCOPE)

- Dataset or CLI flags for choosing an external hardware model path are Phase
  45 scope.
- Score-report warning integration beyond contract compatibility is Phase 45
  scope.
- Structured IR and operator formula work are Phases 42 and 43.
- CDNA 3 / MI300X and CDNA 4 validation remain future milestone work.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HW-01 | Load versioned AMD hardware model JSON artifacts with architecture, dtype/path, peak compute, memory bandwidth, clock policy or assumptions, source, confidence, validation status, and evidence references. [CITED: .planning/REQUIREMENTS.md] | Use a strict hardware artifact model plus loaders for packaged resources and external `Path` JSON. [VERIFIED: codebase grep] |
| HW-02 | Invalid artifacts fail with clear errors for missing provenance, non-positive values, unknown status, or architecture mismatch. [CITED: .planning/REQUIREMENTS.md] | Add focused negative tests for missing provenance, <=0 numeric fields, enum errors, unknown fields, old `validation_status`, and filename/architecture mismatch. [VERIFIED: codebase grep] |
| HW-03 | RDNA 4 `gfx1200` is the only v1.9 validation target; CDNA 3 / MI300X and CDNA 4 remain unvalidated or deferred. [CITED: .planning/REQUIREMENTS.md] | Package only `gfx1200.json` for v1.9 defaults and add guardrails rejecting validated CDNA 3/CDNA 4 claims in model artifacts. [CITED: .planning/ROADMAP.md] |
| HW-04 | Built-in fallback models, if retained, are provisional or unvalidated and use the same validation path as external JSON. [CITED: .planning/REQUIREMENTS.md] | Keep `default_amd_hardware_models()` as a facade over packaged JSON, not embedded constants. [CITED: .planning/phases/41-bound-model-contract-and-hardware-artifacts/41-CONTEXT.md] |
| DOC-01 | Canonical trace JSONL, primary CLI behavior, and public definition/workload/solution schemas remain unchanged. [CITED: .planning/REQUIREMENTS.md] | Extend existing public-contract tests around `Trace.model_dump`, `CliRunner(... --help)`, and schema model dumps. [VERIFIED: codebase grep] |
</phase_requirements>

## Summary

Phase 41 should be implemented as a narrow contract migration in the scoring subsystem: move AMD hardware model constants out of `src/sol_execbench/core/scoring/amd_sol.py` into packaged JSON resources, add strict artifact loading, and keep existing public scoring imports working through a compatibility facade. [VERIFIED: codebase grep] The current code has `AmdHardwareModel` as a frozen dataclass with a single `validation_status` and `default_amd_hardware_models()` returns hard-coded `gfx1200` plus `gfx942` entries, so the highest-risk change is preserving callers while replacing the underlying source of truth. [VERIFIED: codebase grep]

The standard implementation path is stdlib `json`, `pathlib.Path`, `importlib.resources.files(...).joinpath(...).read_text(encoding="utf-8")`, small dataclasses or Pydantic v2 models for strict validation, and pytest guardrails. [CITED: https://docs.python.org/3/library/importlib.resources.html] [CITED: https://pydantic.dev/docs/validation/2.7/api/pydantic/config/] No new runtime dependency is needed. [VERIFIED: pyproject.toml] The package-resource path should be smoke-tested after build or install because `pyproject.toml` currently has no explicit Hatchling file-selection stanza for data resources. [VERIFIED: pyproject.toml] [CITED: https://hatch.pypa.io/1.4/config/build/]

**Primary recommendation:** Add `src/sol_execbench/core/scoring/amd_hardware_models.py` plus `src/sol_execbench/data/amd_hardware_models/gfx1200.json`; make all default and external hardware model loading flow through one strict parser, and update tests before touching later IR or score integration. [VERIFIED: codebase grep]

## Project Constraints (from AGENTS.md)

- Source code lives under `src/sol_execbench/`; the CLI entry point is `sol_execbench.cli:cli`. [CITED: AGENTS.md]
- Tests belong under `tests/sol_execbench/` or `tests/examples/` depending on scope. [CITED: AGENTS.md]
- Use Python 3.12+ and Ruff style; keep focused changes consistent with nearby modules. [CITED: AGENTS.md]
- Pytest is the test framework; use existing environment markers for ROCm and architecture-sensitive tests. [CITED: AGENTS.md]
- Do not commit proprietary kernels, credentials, Hugging Face tokens, downloaded datasets, local cache, build output, or benchmark output. [CITED: AGENTS.md]
- GPU evaluation may require Docker, ROCm hardware, ROCm drivers, `/dev/kfd`, and `/dev/dri`; hardware-specific assumptions should be documented in tests or PR notes. [CITED: AGENTS.md]
- GSD workflow enforcement says repo edits should happen through GSD workflow entry points unless explicitly bypassed. [CITED: AGENTS.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Packaged AMD hardware model defaults | Python package resources | Scoring subsystem | Defaults must be available after install and consumed by scoring code. [CITED: CONTEXT.md] |
| External hardware model path loading | Scoring subsystem API | Filesystem | Phase 41 adds loader support only, not CLI or dataset flags. [CITED: CONTEXT.md] |
| Strict artifact schema validation | Scoring subsystem API | Tests | Unknown fields and invalid values are contract errors before downstream scoring. [CITED: CONTEXT.md] |
| Compatibility API | `core.scoring.amd_sol` facade | New hardware model module | Existing imports such as `default_amd_hardware_models()` should continue deliberately. [CITED: CONTEXT.md] |
| Canonical Trace/CLI/schema guardrails | Tests | Docs | Bound modeling must not mutate canonical benchmark outputs or public schemas. [CITED: REQUIREMENTS.md] |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python stdlib `json` | Python 3.12.13 local | Parse external and packaged JSON artifacts. [VERIFIED: python3 --version] | Already used by local artifact loader patterns. [VERIFIED: codebase grep] |
| Python stdlib `importlib.resources` | Python 3.12 stdlib | Read packaged `src/sol_execbench/data/amd_hardware_models/*.json` resources. [CITED: https://docs.python.org/3/library/importlib.resources.html] | Official docs define `files()` returning `Traversable` resources and `read_text()` for resource content. [CITED: https://docs.python.org/3/library/importlib.resources.html] |
| `pydantic` | 2.12.5 local | Strict schema validation option for hardware artifacts. [VERIFIED: uv run python import] | `ConfigDict(extra='forbid')` rejects unknown fields, matching D-07. [CITED: https://pydantic.dev/docs/validation/2.7/api/pydantic/config/] |
| `pytest` | 9.0.2 local | Unit and contract tests. [VERIFIED: uv run python import] | Existing test suite is pytest with configured markers. [VERIFIED: pyproject.toml] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Hatchling build backend | Declared in `pyproject.toml`; not importable in current `uv run` environment. [VERIFIED: pyproject.toml] [VERIFIED: uv run python import] | Include package data in wheels/sdists. | Add explicit build config or a packaging smoke test if JSON resources are not included by default. [CITED: https://hatch.pypa.io/1.4/config/build/] |
| `slopcheck` | Available at `/home/guohao/.local/bin/slopcheck`. [VERIFIED: which slopcheck] | Package legitimacy checks. | No new external package is recommended, so no package gate is required beyond noting no installs. [VERIFIED: pyproject.toml] |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pydantic model for hardware artifact | Manual dataclass parser with explicit key-set checks | Manual parsing avoids another model layer but is easier to drift; Pydantic `extra='forbid'` directly supports unknown-field rejection. [CITED: https://pydantic.dev/docs/validation/2.7/api/pydantic/config/] |
| `importlib.resources.files()` | `Path(__file__).parent` | `__file__` path logic is less robust for installed package resources and zip-style loaders; official resource APIs are designed for package resources. [CITED: https://docs.python.org/3/library/importlib.resources.html] |
| Keep hard-coded constants | JSON resource source of truth | Hard-coded constants violate D-09 and make packaged artifact validation impossible. [CITED: CONTEXT.md] |

**Installation:**

```bash
# No new external packages are recommended for Phase 41.
uv sync --all-groups
```

**Version verification performed:**

```bash
python3 --version
uv --version
uv run python -c "import pydantic, pytest; print(pydantic.__version__, pytest.__version__)"
```

## Package Legitimacy Audit

No new external packages are recommended or installed in Phase 41. [VERIFIED: pyproject.toml] Package legitimacy gate is not applicable because the planner should use existing stdlib, Pydantic, pytest, and Hatchling declarations. [VERIFIED: pyproject.toml]

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| None | - | - | - | - | - | No new package install |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```text
Packaged resource JSON
  src/sol_execbench/data/amd_hardware_models/gfx1200.json
        |
        v
importlib.resources.files("sol_execbench.data.amd_hardware_models")
        |
        v
load_packaged_amd_hardware_model("gfx1200")
        |
        v
strict parser: schema_version, architecture, dtype_or_path, peaks, bandwidth,
source/provenance, confidence, hardware_validation_status, model_validation_status
        |
        +--> clear ValueError on missing provenance, <=0 values, unknown fields,
        |    unknown status, old validation_status, architecture mismatch
        |
        v
AmdHardwareModel compatibility dataclass
        |
        +--> default_amd_hardware_models()["gfx1200"]
        |
        v
build_amd_sol_bound_artifact(...)
        |
        v
derived AMD SOL artifact only; Trace JSONL, CLI help, Definition/Workload/Solution schemas unchanged
```

### Recommended Project Structure

```text
src/sol_execbench/
├── data/
│   ├── __init__.py
│   └── amd_hardware_models/
│       ├── __init__.py
│       └── gfx1200.json
└── core/scoring/
    ├── amd_hardware_models.py
    └── amd_sol.py

tests/sol_execbench/
├── test_amd_hardware_models.py
├── test_amd_sol_bounds.py
└── test_public_contract_guardrails.py
```

### Pattern 1: Single Strict Loader For Packaged And External JSON

**What:** Normalize all hardware model JSON payloads through one parser function, then have packaged-resource and external-path loaders only differ in how they read text. [VERIFIED: codebase grep]

**When to use:** Use for `load_amd_hardware_model(path)`, `load_packaged_amd_hardware_model(architecture)`, and `default_amd_hardware_models()`. [CITED: CONTEXT.md]

**Example:**

```python
# Source: Python importlib.resources docs and Pydantic config docs.
from importlib import resources
from pathlib import Path
from typing import Any
import json

from pydantic import BaseModel, ConfigDict, Field


class AmdHardwareModelArtifactV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    architecture: str
    dtype_or_path: str
    peak_tflops: float = Field(gt=0)
    memory_bandwidth_gbps: float = Field(gt=0)
    source: str = Field(min_length=1)
    confidence: str
    hardware_validation_status: str
    model_validation_status: str
    evidence_refs: list[str] = Field(default_factory=list)


def load_packaged_amd_hardware_model(architecture: str) -> AmdHardwareModel:
    package = "sol_execbench.data.amd_hardware_models"
    resource = resources.files(package).joinpath(f"{architecture}.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return amd_hardware_model_from_dict(payload, source=f"package:{architecture}.json")


def load_amd_hardware_model(path: Path) -> AmdHardwareModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return amd_hardware_model_from_dict(payload, source=str(path))
```

### Pattern 2: Compatibility Facade In `amd_sol.py`

**What:** Keep public names imported from `amd_sol.py`, but move hardware artifact parsing into a focused module. [VERIFIED: codebase grep]

**When to use:** Use when preserving existing tests and imports around `AmdHardwareModel`, `HardwareValidationStatus`, and `default_amd_hardware_models()`. [VERIFIED: codebase grep]

**Example:**

```python
# Source: existing amd_sol.py facade surface.
def default_amd_hardware_models() -> dict[str, AmdHardwareModel]:
    """Return packaged AMD hardware model entries."""
    model = load_packaged_amd_hardware_model("gfx1200")
    return {model.architecture: model}
```

### Pattern 3: Contract Tests Before Integration

**What:** Add tests proving new derived artifacts do not affect canonical traces, public schema model dumps, or primary CLI help. [VERIFIED: codebase grep]

**When to use:** Use for DOC-01 and before any later Phase 42-45 estimator/report changes. [CITED: REQUIREMENTS.md]

**Example:**

```python
# Source: tests/sol_execbench/test_public_contract_guardrails.py pattern.
before = trace.model_dump(mode="json")
_ = build_amd_sol_bound_artifact(definition, trace.workload, hardware).to_dict()
assert trace.model_dump(mode="json") == before
```

### Anti-Patterns to Avoid

- **Dual parser paths:** Do not parse packaged JSON one way and external JSON another way; that would violate HW-04's same-validation-path requirement. [CITED: REQUIREMENTS.md]
- **Silent compatibility fallback:** Do not catch invalid packaged JSON and return embedded constants; D-08 and D-10 require explicit failure. [CITED: CONTEXT.md]
- **Leaving `validation_status` in v2 JSON:** v2 artifacts must use `hardware_validation_status` and `model_validation_status` only. [CITED: CONTEXT.md]
- **Adding CLI flags now:** External JSON path support is an API foundation in Phase 41; dataset/CLI wiring is deferred. [CITED: CONTEXT.md]
- **Validating CDNA 3/CDNA 4 by implication:** Existing solution schema supports `gfx940`, `gfx941`, and `gfx942`, but Phase 41 model artifacts must not claim those as v1.9 validated targets. [VERIFIED: codebase grep] [CITED: REQUIREMENTS.md]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Package resource loading | `__file__` path joins and installed-location assumptions | `importlib.resources.files(...).joinpath(...).read_text()` | Official API supports package resources independent of direct filesystem layout. [CITED: https://docs.python.org/3/library/importlib.resources.html] |
| Unknown-field rejection | Ad hoc `for key in payload` checks in multiple loaders | Pydantic v2 `ConfigDict(extra='forbid')` or one centralized key-set check | Official Pydantic config directly rejects extras. [CITED: https://pydantic.dev/docs/validation/2.7/api/pydantic/config/] |
| Artifact provenance validation | Free-form undocumented strings only | Required `source` plus optional `evidence_refs` and clock policy/assumptions fields | HW-01/HW-02 require provenance and evidence references. [CITED: REQUIREMENTS.md] |
| Public contract verification | Manual reviewer memory | Existing pytest guardrail tests | Current tests already assert trace immutability, CLI help stability, and schema dumps. [VERIFIED: codebase grep] |

**Key insight:** The hardware model is a public contract input to derived scoring, not a hidden implementation constant. Treating JSON artifacts as the source of truth keeps provenance, validation state, and future model changes reviewable. [CITED: REQUIREMENTS.md]

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|-----------------|
| Stored data | None. No database, Redis, Chroma, or persistent scoring artifact store was found in phase scope; current model constants live in source code. [VERIFIED: codebase grep] | No data migration. |
| Live service config | None. No live service configuration owns hardware model constants for Phase 41. [VERIFIED: codebase grep] | No service update. |
| OS-registered state | None. No systemd/pm2/launchd/task registration is part of this package contract phase. [VERIFIED: codebase grep] | No OS registration update. |
| Secrets/env vars | None. Hardware model loading should not add secrets or env vars. [CITED: REQUIREMENTS.md] | No secret migration. |
| Build artifacts | Existing installed wheels may not include new JSON resources until rebuilt/reinstalled; Hatchling is declared but not importable in the current runtime environment. [VERIFIED: pyproject.toml] [VERIFIED: uv run python import] | Add packaging/importlib resource smoke test; run `uv sync --all-groups` or build/install before release validation. |

## Common Pitfalls

### Pitfall 1: Packaged JSON Not Included In Built Wheel

**What goes wrong:** Tests pass from the source tree, but an installed package cannot load `gfx1200.json`. [ASSUMED]
**Why it happens:** Build backends apply file-selection rules; `pyproject.toml` has no explicit Hatchling data-resource include stanza today. [VERIFIED: pyproject.toml] [CITED: https://hatch.pypa.io/1.4/config/build/]
**How to avoid:** Add a packaging smoke test that imports the installed package or builds a wheel and checks `importlib.resources.files("sol_execbench.data.amd_hardware_models").joinpath("gfx1200.json").is_file()`. [CITED: https://docs.python.org/3/library/importlib.resources.html]
**Warning signs:** `FileNotFoundError`, empty `default_amd_hardware_models()`, or tests that only use direct `Path("src/...")`. [ASSUMED]

### Pitfall 2: V1 And V2 Status Fields Coexist

**What goes wrong:** JSON accepts both `validation_status` and the new split fields, creating ambiguous downstream semantics. [CITED: CONTEXT.md]
**Why it happens:** Backward compatibility pressure around `AmdHardwareModel.validation_status`. [VERIFIED: codebase grep]
**How to avoid:** Reject `validation_status` in v2 JSON through `extra='forbid'` and adapt only inside legacy compatibility code if needed. [CITED: https://pydantic.dev/docs/validation/2.7/api/pydantic/config/] [CITED: CONTEXT.md]
**Warning signs:** Tests asserting `payload["hardware_model"]["validation_status"]` for v2 artifacts without updated semantics. [VERIFIED: codebase grep]

### Pitfall 3: Architecture Validation Is Too Permissive

**What goes wrong:** A `gfx942` or CDNA 4 model enters defaults or gets marked validated because the enum permits it elsewhere. [ASSUMED]
**Why it happens:** `SupportedHardware` already includes CDNA 3 schema targets, but model validation scope is narrower than solution schema support. [VERIFIED: codebase grep] [CITED: REQUIREMENTS.md]
**How to avoid:** Phase 41 defaults should contain only `gfx1200`; if parser supports arbitrary architectures for external artifacts, validation status must reject validated CDNA 3/CDNA 4 claims for v1.9. [CITED: CONTEXT.md]
**Warning signs:** `default_amd_hardware_models()["gfx942"]` still exists, or docs imply MI300X/CDNA3 model validation. [VERIFIED: codebase grep]

### Pitfall 4: Public Contract Tests Are Updated To Match Drift

**What goes wrong:** New bound fields leak into trace JSONL, CLI help, or public schemas, and tests get changed to accept it. [ASSUMED]
**Why it happens:** Derived artifact work is adjacent to reporting code and easy to confuse with canonical outputs. [CITED: .planning/research/SUMMARY.md]
**How to avoid:** Treat `Trace`, `Definition`, `Workload`, `Solution`, and primary `sol-execbench --help` snapshots as negative guardrails; add separate tests for derived artifacts. [VERIFIED: codebase grep]
**Warning signs:** `--sol-bound`, `--amd-score-report`, or hardware model path flags appear in primary CLI help during Phase 41. [VERIFIED: codebase grep] [CITED: CONTEXT.md]

## Code Examples

Verified patterns from official and local sources:

### Read Packaged Resource Text

```python
# Source: https://docs.python.org/3/library/importlib.resources.html
from importlib import resources

resource = resources.files("sol_execbench.data.amd_hardware_models").joinpath("gfx1200.json")
payload_text = resource.read_text(encoding="utf-8")
```

### Reject Unknown JSON Fields With Pydantic

```python
# Source: https://pydantic.dev/docs/validation/2.7/api/pydantic/config/
from pydantic import BaseModel, ConfigDict


class HardwareArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    architecture: str
```

### Keep Compatibility API Loading Packaged Defaults

```python
# Source: src/sol_execbench/core/scoring/amd_sol.py existing public function name.
def default_amd_hardware_models() -> dict[str, AmdHardwareModel]:
    hardware = load_packaged_amd_hardware_model("gfx1200")
    return {hardware.architecture: hardware}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hard-coded `default_amd_hardware_models()` constants | Versioned packaged JSON artifacts loaded through `importlib.resources` | Phase 41 planned | Makes provenance and validation status auditable. [CITED: CONTEXT.md] |
| Single `validation_status` | Split `hardware_validation_status` and `model_validation_status` | Phase 41 planned | Separates environment validation from bound-model validation. [CITED: CONTEXT.md] |
| Derived AMD SOL v1 hardware dict in artifact payload | v2-compatible hardware model contract foundation | Phase 41 planned | Later phases can consume stable model refs without mutating trace JSONL. [CITED: ROADMAP.md] |

**Deprecated/outdated:**

- `validation_status` in v2 hardware model JSON is deprecated by decision and must be rejected. [CITED: CONTEXT.md]
- Hard-coded peak/bandwidth/source metadata in `default_amd_hardware_models()` is deprecated by decision and must move to JSON. [CITED: CONTEXT.md]
- Built-in `gfx942` default hardware model is incompatible with Phase 41 success criteria unless removed from defaults or kept only as explicitly unvalidated external/test data outside v1.9 packaged defaults. [CITED: ROADMAP.md] [VERIFIED: codebase grep]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Source-tree tests can pass while a built wheel omits JSON package data. | Common Pitfalls | Planner may skip packaging smoke tests and ship missing defaults. |
| A2 | `FileNotFoundError` or direct `Path("src/...")` use are warning signs for package-resource drift. | Common Pitfalls | Planner may choose too narrow tests. |
| A3 | Architecture validation may become too permissive because schema-supported hardware targets differ from model-validated targets. | Common Pitfalls | Planner may accidentally keep `gfx942` as a default or mark CDNA 3 validated. |
| A4 | Public contract tests may be weakened to match drift during derived artifact work. | Common Pitfalls | Canonical trace or CLI behavior could change unnoticed. |

## Open Questions (RESOLVED)

1. **RESOLVED: Should `AmdHardwareModel` expose both split statuses or keep a derived compatibility `validation_status` property?**
   - What we know: v2 JSON must use split fields only, but existing tests and score code reference `hardware_model.validation_status`. [VERIFIED: codebase grep] [CITED: CONTEXT.md]
   - Decision: Add split dataclass fields as the canonical public object state. Do not include `validation_status` in v2 JSON or `to_dict()` output. A derived compatibility property may exist only if required to keep existing internal score code working during Phase 41, and it must not serialize into v2 artifacts. [CITED: CONTEXT.md]

2. **RESOLVED: Should external artifacts for non-`gfx1200` be loadable but unvalidated, or rejected in Phase 41?**
   - What we know: Loader support for arbitrary external JSON path is required, and RDNA 4 `gfx1200` is the only v1.9 validation target. [CITED: CONTEXT.md]
   - Decision: Allow external non-`gfx1200` artifacts only when both status fields are non-`validated`; keep packaged defaults to `gfx1200` only. Reject non-`gfx1200` artifacts that claim v1.9 validation. [CITED: CONTEXT.md]

3. **RESOLVED: Does Hatchling include `src/sol_execbench/data/**/*.json` by default for this project?**
   - What we know: Hatchling is the build backend and package-resource docs require resources to be packaged. [VERIFIED: pyproject.toml] [CITED: https://docs.python.org/3/library/importlib.resources.html]
   - Decision: Treat installed-package resource availability as mandatory execution verification. Plan 41-01 must include either a build/install resource smoke test or an explicit Hatchling include stanza plus artifact-content verification. [CITED: https://hatch.pypa.io/1.4/config/build/]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| Python | Runtime and importlib.resources | yes | 3.12.13 | None needed. [VERIFIED: python3 --version] |
| uv | Test and dependency commands | yes | 0.11.15 | System Python for narrow unit tests. [VERIFIED: uv --version] |
| pydantic | Optional strict artifact model | yes | 2.12.5 | Manual centralized parser, but Pydantic is already installed. [VERIFIED: uv run python import] |
| pytest | Validation | yes | 9.0.2 | None needed. [VERIFIED: uv run python import] |
| Hatchling | Build backend for packaging resource smoke | declared, not importable in runtime env | not available via `uv run python -c "import hatchling"` | Use `uv build` or `uv sync --all-groups` to provision build backend during implementation. [VERIFIED: pyproject.toml] |
| slopcheck | Package legitimacy audit if new package added | yes | path found | Not needed because no new packages. [VERIFIED: which slopcheck] |

**Missing dependencies with no fallback:**
- None for code/test implementation. [VERIFIED: environment audit]

**Missing dependencies with fallback:**
- Hatchling is not importable in the current `uv run` environment, but the build backend is declared and can be exercised through packaging/build commands during implementation. [VERIFIED: pyproject.toml]

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 [VERIFIED: uv run python import] |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` [VERIFIED: pyproject.toml] |
| Quick run command | `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x` |
| Full suite command | `uv run pytest tests/` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| HW-01 | Packaged and external v2 hardware JSON load with required fields. | unit | `uv run pytest tests/sol_execbench/test_amd_hardware_models.py -x` | No, Wave 0 |
| HW-02 | Invalid provenance, values, status, unknown fields, old `validation_status`, and architecture mismatch raise clear errors. | unit | `uv run pytest tests/sol_execbench/test_amd_hardware_models.py -x` | No, Wave 0 |
| HW-03 | Defaults include only `gfx1200`; CDNA 3/CDNA 4 cannot be marked v1.9 validated. | unit/docs guardrail | `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_public_contract_guardrails.py -x` | Partial |
| HW-04 | `default_amd_hardware_models()` loads packaged JSON and fails on invalid/missing package data. | unit | `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_sol_bounds.py -x` | Partial |
| DOC-01 | Trace JSONL, CLI help, Definition/Workload/Solution schemas remain unchanged. | contract | `uv run pytest tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_amd_sol_bounds.py -x` | Yes, needs extension |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/sol_execbench/test_amd_hardware_models.py tests/sol_execbench/test_amd_sol_bounds.py tests/sol_execbench/test_public_contract_guardrails.py -x`
- **Per wave merge:** `uv run pytest tests/sol_execbench/test_amd_native_score.py tests/sol_execbench/test_run_dataset_amd_score.py tests/sol_execbench/test_public_contract_guardrails.py -x`
- **Phase gate:** `uv run pytest tests/`

### Wave 0 Gaps

- [ ] `tests/sol_execbench/test_amd_hardware_models.py` - covers HW-01 through HW-04 strict loader behavior. [VERIFIED: tests listing]
- [ ] Package-resource smoke test for `sol_execbench.data.amd_hardware_models.gfx1200.json`. [CITED: https://docs.python.org/3/library/importlib.resources.html]
- [ ] Public-contract guardrail extension for v1.9 no-claim terms and no new primary CLI options. [VERIFIED: codebase grep]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|------------------|
| V2 Authentication | no | No authentication surface in Phase 41. [CITED: REQUIREMENTS.md] |
| V3 Session Management | no | No session surface in Phase 41. [CITED: REQUIREMENTS.md] |
| V4 Access Control | no | Local file loading only; no authorization boundary. [CITED: REQUIREMENTS.md] |
| V5 Input Validation | yes | Strict JSON validation, enum validation, positive numeric constraints, architecture/status policy checks. [CITED: REQUIREMENTS.md] |
| V6 Cryptography | no | No cryptographic operation in Phase 41. [CITED: REQUIREMENTS.md] |

### Known Threat Patterns for Python JSON Artifact Loading

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed or schema-drifted hardware model JSON | Tampering | `extra='forbid'`, required provenance, enum checks, positive numeric constraints. [CITED: https://pydantic.dev/docs/validation/2.7/api/pydantic/config/] |
| Misleading validation claims | Repudiation | Explicit split validation status fields and no-claim guardrail tests. [CITED: CONTEXT.md] |
| Path traversal through package resource name | Tampering | Loader should accept architecture tokens, not arbitrary resource path fragments; external path loader should use explicit `Path` API. [CITED: https://docs.python.org/3/library/importlib.resources.html] |
| Canonical output mutation | Tampering | Tests compare `Trace.model_dump(mode="json")` before/after derived artifact generation. [VERIFIED: codebase grep] |

## Sources

### Primary (HIGH confidence)

- `AGENTS.md` - project structure, commands, style, testing, security, and GSD constraints.
- `.planning/phases/41-bound-model-contract-and-hardware-artifacts/41-CONTEXT.md` - locked implementation decisions.
- `.planning/REQUIREMENTS.md` - HW-01 through HW-04 and DOC-01 requirements.
- `.planning/ROADMAP.md` - Phase 41 success criteria and phase boundaries.
- `.planning/research/SUMMARY.md` - v1.9 architecture baseline and guardrails.
- `src/sol_execbench/core/scoring/amd_sol.py` - current hardware model dataclass, hard-coded defaults, bound artifact shape.
- `src/sol_execbench/core/scoring/baseline_artifact.py` - local artifact loader error style.
- `tests/sol_execbench/test_amd_sol_bounds.py` - current AMD SOL artifact and trace immutability tests.
- `tests/sol_execbench/test_public_contract_guardrails.py` - current CLI/schema/claim guardrail tests.
- Python docs: https://docs.python.org/3/library/importlib.resources.html - package resource loading API.
- Pydantic docs: https://pydantic.dev/docs/validation/2.7/api/pydantic/config/ - `extra='forbid'` behavior.

### Secondary (MEDIUM confidence)

- Hatch build docs: https://hatch.pypa.io/1.4/config/build/ - file selection and package inclusion configuration.

### Tertiary (LOW confidence)

- None used as authoritative support.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH - all recommended implementation tools are existing dependencies or Python stdlib, with official docs checked. [VERIFIED: pyproject.toml] [CITED: Python/Pydantic docs]
- Architecture: HIGH - phase context and current code point to a narrow scoring-subsystem contract migration. [CITED: CONTEXT.md] [VERIFIED: codebase grep]
- Pitfalls: MEDIUM - public-contract and status-field risks are verified in code/context; package-data omission risk needs implementation-time smoke verification. [VERIFIED: codebase grep] [ASSUMED]

**Research date:** 2026-05-23
**Valid until:** 2026-06-22 for project-local architecture; re-check official packaging/Pydantic docs if implementation happens after dependency updates.
