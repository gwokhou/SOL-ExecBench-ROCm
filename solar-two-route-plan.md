# SOLAR two-route architecture

Status: implemented and verified.

## Contract

- `DEFAULT_ROUTE` is `Route.NVLABS`.
- `DEFAULT_IR_KIND` is `IRKind.NVLABS_EINSUM`.
- Both routes target NVLabs einsum by default.
- `Route.NVLABS` captures an operator graph with torchview and supports
  `NVLABS_EINSUM`.
- `Route.MAINLINE` captures an operator graph with make_fx and defaults to
  `NVLABS_EINSUM`; callers may explicitly request the retained ATen IR.
- The retired IR and its operation payload have no production compatibility
  path.

## Shared boundaries

`RouteSpec` declaratively maps each route to an `ExtractionKind`. Supported
input graph kinds belong to `IRBackend`, so the route layer has no dependency
on IR backend capabilities. Public pipeline and readiness code do not import
concrete route implementations.

```text
AnalysisRequest / ConversionReadinessRequest
                 |
                 v
          shared workflow stages
                 |
       +---------+----------+
       |                    |
 route_spec().extraction   IRKind
       |                    |
 graph registry          IR registry
       |                    |
 make_fx / torchview     conversion, verification, analysis
```

- `solar.graph.extraction.extract_operator_graph` dispatches exclusively
  through the graph backend registry.
- `solar.ir.conversion.convert_operator_graph` validates registered extraction
  provenance and dispatches exclusively through the IR backend registry.
- Every IR backend returns an `IRGraphArtifact`; every persisted IR graph must
  carry an explicit `ir_kind`.
- `solar.pipeline` and `solar.readiness` reuse the same extraction, conversion,
  and verification workflow functions.
- Verification and analysis select implementations from the resulting
  `IRGraphArtifact.kind`, not from the originating route.

## Naming policy

Public/shared identifiers describe lifecycle or semantic roles:
`GraphBackend`, `IRBackend`, `RouteSpec`, `WorkflowRequest`,
`IRGraphArtifact`, `LayerContractionAnalysis`, and `layer_operation`.
Backend and route names occur only as explicit discriminators or private
adapter names.

The shared semantic operation fallback uses `OPERATION_KIND`; contraction
classification uses `CONTRACTION_KIND`. General operation semantics live in
`semantic_op`.

## Regression gates

- Public pipeline/readiness modules may not import concrete route
  implementations.
- Public common identifiers may not contain concrete backend or route names,
  except discriminator enums.
- Common extraction/conversion dispatch may not use route-specific fallback
  branches.
- Both extraction routes must produce equivalent NVLabs contraction semantics
  for the shared matmul reference.
- Extraction/representation incompatibilities fail at the shared IR conversion
  boundary.
- IR execution rejects graphs without an explicit `ir_kind`.

Verification commands:

```text
uv run pytest tests/
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
uv run ty check
uv run --no-sync python scripts/check_coupling.py
uv run --no-sync python scripts/check_readability.py
uv run --no-sync python scripts/check_production_reachability.py
```
