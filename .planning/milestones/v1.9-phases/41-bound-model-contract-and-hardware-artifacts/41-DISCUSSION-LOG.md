# Phase 41: Bound Model Contract And Hardware Artifacts - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-23
**Phase:** 41-Bound Model Contract And Hardware Artifacts
**Areas discussed:** Artifact location and packaging, Hardware validation status semantics, Artifact schema strictness, Fallback model policy, Public-contract guardrail depth

---

## Artifact Location And Packaging

| Option | Description | Selected |
|--------|-------------|----------|
| Packaged source data | Put default hardware model JSON under `src/sol_execbench/data/amd_hardware_models/` and load it with package resources. | ✓ |
| Repo `data/` artifact | Treat hardware models as repository data artifacts outside packaged source. | |
| Tests first only | Add only fixtures first and defer runtime packaged defaults. | |
| Hybrid | Packaged default plus immediate external artifact integration. | |

**User's choice:** Packaged source data.
**Notes:** Default `gfx1200` hardware model should be available after package installation.

| Option | Description | Selected |
|--------|-------------|----------|
| Loader-only override | Implement loader support for arbitrary JSON path, but defer dataset/CLI flags. | ✓ |
| Dataset flag now | Add user-facing dataset hardware model option in Phase 41. | |
| No override yet | Support only packaged defaults for now. | |

**User's choice:** Loader-only override.
**Notes:** Dataset and CLI integration remains Phase 45 scope.

---

## Hardware Validation Status Semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Provisional until Phase 46 | Keep `gfx1200` fully provisional until validation closure. | |
| Validated contract now | Allow `gfx1200` artifact to be marked validated immediately. | |
| Dual status fields | Separate hardware/environment validation from bound-model validation. | ✓ |

**User's choice:** Dual status fields.
**Notes:** Avoid conflating prior RDNA 4 validation with validation of the new bound-model contract.

| Option | Description | Selected |
|--------|-------------|----------|
| Additive migration | Keep `validation_status` and add two new fields. | |
| Hard replace | v2 contract replaces `validation_status` with `hardware_validation_status` and `model_validation_status`. | ✓ |
| Nested validation object | Put statuses under a nested `validation` object. | |

**User's choice:** Hard replace.
**Notes:** Legacy compatibility can translate internally, but v2 JSON must not rely on the old single field.

---

## Artifact Schema Strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Strict reject unknown fields | Unknown fields fail validation. | ✓ |
| Permissive with warning | Unknown fields are accepted with warnings. | |
| Strict packaged, permissive external | Built-ins are strict; external user artifacts are loose. | |

**User's choice:** Strict reject unknown fields.
**Notes:** Phase 41 is contract-setting work, so schema drift should fail early.

---

## Fallback Model Policy

| Option | Description | Selected |
|--------|-------------|----------|
| Compatibility wrapper over JSON | Keep `default_amd_hardware_models()` but make it load packaged JSON. | |
| Deprecate but keep hard-coded | Add JSON loader while keeping hard-coded fallback constants temporarily. | |
| Remove hard-coded fallback | No embedded hardware constants; packaged JSON is required. | ✓ |

**User's choice:** Remove hard-coded fallback.
**Notes:** `default_amd_hardware_models()` may remain for compatibility, but it must load packaged JSON and fail explicitly if packaged data is missing or invalid.

---

## Public-Contract Guardrail Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Runtime contract only | Guard only trace/CLI/schema immutability. | |
| Runtime + claim docs | Also guard docs/claims against overclaiming. | ✓ |
| Full guardrail now | Add score-report/schema warning guardrails immediately too. | |

**User's choice:** Runtime + claim docs.
**Notes:** Phase 41 should block B200/SOLAR/leaderboard/CDNA3/MI300X/CDNA4 overclaims early, but full score integration remains Phase 45.

---

## the agent's Discretion

- Exact module naming, dataclass/Pydantic split, and test file organization are left to implementation planning.
- Compatibility details are flexible as long as hard-coded hardware model values are removed from the fallback path and public contracts stay stable.

## Deferred Ideas

- Dataset or CLI hardware-model selection flags belong to Phase 45.
- Full score warning integration belongs to Phase 45 unless compatibility requires a small shim.
- Structured graph IR and operator estimator work belong to Phases 42 and 43.
