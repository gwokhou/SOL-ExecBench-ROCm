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

Business logic must not add new `isinstance(value, dict)` payload checks. Use
`sol_execbench.core.data.path_access` helpers or define a local typed adapter.
