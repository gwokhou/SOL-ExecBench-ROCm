# Phase 42: Structured Bound Graph IR - Patterns

**Mapped:** 2026-05-23
**Status:** Ready for planning

## File Pattern Map

| Target File | Role | Closest Existing Analog | Reuse Pattern |
| --- | --- | --- | --- |
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | New graph IR contract and extractor | `src/sol_execbench/core/scoring/amd_sol.py` | Frozen dataclasses, `to_dict()` methods, small pure helpers, explicit confidence labels. |
| `src/sol_execbench/core/scoring/amd_sol.py` | Compatibility facade | Existing same file | Keep public functions and dataclasses import-compatible; delegate new internals behind existing APIs. |
| `src/sol_execbench/core/scoring/__init__.py` | Public scoring export surface | Existing same file | Export new graph types deliberately, update `__all__`, avoid accidental schema exposure. |
| `tests/sol_execbench/test_amd_bound_graph.py` | New IR unit tests | `tests/sol_execbench/test_amd_sol_bounds.py` | Build small `Definition` and `Workload` fixtures inline, assert exact graph metadata. |
| `tests/sol_execbench/test_amd_sol_bounds.py` | Compatibility tests | Existing same file | Preserve current behavior while adding assertions that facade nodes come from the new graph path. |
| `tests/sol_execbench/test_public_contract_guardrails.py` | Schema/CLI/Trace guardrails | Existing same file | Assert derived graph APIs do not add fields to canonical schemas or primary CLI help. |

## Existing Code Excerpts To Reuse

### Dataclass Serialization

`amd_sol.py` uses frozen dataclasses with direct JSON-like dict conversion:

```python
@dataclass(frozen=True)
class GraphNode:
    node_id: str
    op_type: str
    expression: str
    confidence: EstimateConfidence
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "op_type": self.op_type,
            "expression": self.expression,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
        }
```

Use the same style for `BoundGraph`, `BoundGraphNode`, `BoundTensor`, and edge
records.

### Shape Resolution

`Definition` already resolves workload-bound tensor shapes:

```python
input_shapes = definition.get_input_shapes(workload.axes)
output_shapes = definition.get_output_shapes(workload.axes)
```

The new graph builder should reuse these helpers rather than reimplementing
symbolic axis resolution.

### Compatibility Facade

Current public callers use:

```python
graph_nodes = extract_graph(definition)
work_estimates = estimate_work(definition, workload, graph_nodes)
artifact = build_amd_sol_bound_artifact(definition, workload, hardware_model)
```

Phase 42 must preserve that path. Add a new workload-aware graph API without
breaking the old one.

## Data Flow Target

```text
Definition + Workload
  -> resolved input/output BoundTensor records
  -> isolated reference tracing path
  -> BoundGraphNode records with inputs/outputs/source/confidence
  -> producer/consumer edges
  -> BoundGraph.to_dict()
  -> amd_sol.extract_graph() facade flattening for legacy GraphNode users
```

## Testing Pattern

Use fixtures like the existing tests:

```python
definition = Definition(
    name="projection_residual",
    axes={...},
    inputs={...},
    outputs={...},
    reference="import torch\n\ndef run(...):\n    ...",
)
workload = Workload(axes={...}, inputs={...}, uuid="...")
```

Then assert exact values:

- `graph.nodes[0].node_id == "op_1"`
- `graph.nodes[0].op_family == "gemm"` or `"linear_projection"`
- `graph.tensors["input:attn_output"].shape == (16, 512, 2560)`
- unsupported calls have `confidence == EstimateConfidence.UNSUPPORTED`
- tuple outputs produce distinct output tensor roles.

