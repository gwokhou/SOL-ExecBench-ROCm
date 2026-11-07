# Schema Boundaries

Raw JSON dictionaries are allowed only at external artifact boundaries. Boundary
code must convert payloads into Pydantic models, frozen dataclasses, or named view
adapters before report assembly, scoring, claim, or consistency logic consumes the
data.

Allowed raw payload infrastructure:

- `sol_execbench.core.data.json_utils`
- `sol_execbench.core.data.solution`

Allowed artifact boundaries:

- benchmark artifact readers
- current, version-specific sidecar readers

Allowed parser boundaries:

- `*_parsing.py` modules
- source reference adapters such as `paper_denominator_sources.py`
- sidecar readers that require their sole supported schema version

Business logic must not add new `isinstance(value, dict)` payload checks.
Define a focused typed adapter at the artifact parser boundary instead of
adding generic nested-payload access helpers.

The canonical benchmark inputs are versioned wire contracts:

- `sol_execbench.definition.v1`
- `sol_execbench.workload.v2`
- `sol_execbench.solution.v1`
- `sol_execbench.benchmark_config.v2`
- `sol_execbench.trace.v1`

Every canonical artifact instance must carry its exact `schema_version`.
Readers reject missing, retired, or future versions before validating business
fields.
Repository manifests, readiness matrices, diagnostic audits, CLI envelopes,
and SOLAR request manifests follow the same rule through their own strict
nested models.

Registration follows the serialized artifact boundary, not Python class
visibility. A public class that is used only in memory does not gain a schema
solely because it is importable. Nested models are covered by the versioned
top-level envelope unless they are also stored or exchanged independently.
Variants that share one storage and reader boundary use one schema family plus
an explicit discriminator such as `stage`, `role`, or `artifact_kind`. In
particular, lifecycle stage manifests share
`sol_execbench.diagnostic_lifecycle_manifest.v1`, and release execution plans,
run statements, environments, and SOLAR indexes share
`sol_execbench.release_component.v1`. Lifecycle state
objects share `sol_execbench.diagnostic_lifecycle_state.v1`; corpus readiness
records and summaries share `sol_execbench.corpus_readiness.v1`; platform
preflight variants share `sol_execbench.platform_preflight.v1`; qualification
gates and receipts share their owning qualification contract; and each release
packager uses one package contract for its archive and attestation variants.
The same rule covers CLI contract/response variants, environment
snapshot/diagnostic variants, performance evidence components, diagnostic
acceptance and case-reuse variants, corpus design/preflight variants, and
content-addressed release components.

Canonical registrations live with their owning domain:

- benchmark inputs in `core/data/schema_versions.py`
- datasets in `core/dataset/schema_versions.py`
- platform evidence in `core/platform/schema_versions.py`
- execution control in `core/control_plane_schema_versions.py`
- profiler, performance, diagnostics, and lifecycle contracts beside their
  respective benchmark packages
- release and scoring artifacts in `core/scoring/schema_versions.py`

Producers and readers import their domain enum directly. The integrity-layer
`artifact_registry.py` combines the registries only for repository-wide audits;
it is not a public schema namespace or an import path for business logic.

Rebuildable caches are not canonical artifacts. They use a local
`format_version`, include that value in their cache identity, and refresh on a
format mismatch. Versioned behavior selectors and transport protocols likewise
live in the wire-protocol registry rather than artifact domain registries; they
may appear inside a versioned artifact without defining a second artifact
schema family.
Internal checksum preimages follow the same local-format rule. If their digest
is exposed by a canonical artifact, changing the preimage format also bumps the
enclosing artifact schema.

## Non-canonical local evidence

`data/calibration/` and `data/local-evidence/` hold gfx1200 diagnostic evidence
left over from local ROCm port work. They are git-ignored (`/data/*`),
unreferenced by `src/`, `scripts/`, or `tests/`, and carry schema identifiers
(under the `sol_execbench.` namespace) absent from the canonical domain
registries:

- `hardware_calibration` v1 and v2
- `amd_hardware_model` v3
- `fusion_validation` v1
- `hardware_profile_requirements` v1
- `representative_suite` v1

Governance rule: these are non-canonical local evidence, not
corpus. Their identifiers must **not** be registered, and the files must **not**
be wired into evaluation or SOLAR consumption. A `NON_CANONICAL.md` marker in
each directory records the same decision on disk.
