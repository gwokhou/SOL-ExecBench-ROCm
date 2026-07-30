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
- `sol_execbench.workload.v1`
- `sol_execbench.solution.v1`
- `sol_execbench.benchmark_config.v1`
- `sol_execbench.trace.v1`

Every persisted instance must carry its exact `schema_version`. Readers reject
missing, retired, or future versions before validating business fields.
Repository manifests, readiness matrices, diagnostic audits, CLI envelopes,
and SOLAR request manifests follow the same rule through their own strict
nested models.
