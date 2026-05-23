# Technology Stack

**Project:** SOL ExecBench ROCm Port
**Milestone:** v1.10 paper-aligned SOLAR automatic derivation
**Researched:** 2026-05-23
**Scope:** SOLAR derivation only. No 124-model / 235-problem extraction, no new real-hardware validation, no hosted leaderboard.
**Overall confidence:** HIGH for repository integration points; MEDIUM for exact per-family formulas until phase golden cases are selected.

## Recommendation

Do not add a new framework dependency. v1.10 should turn the current AMD SOL/SOLAR path into an automatic derivation system by extending the existing stack:

| Layer | Keep / Add | Decision |
|-------|------------|----------|
| Source extraction | Keep stdlib `ast`; keep best-effort `torch.fx.symbolic_trace` + `ShapeProp` | Use FX for traceable PyTorch references and tensor metadata, then use AST as deterministic fallback and source-of-truth audit trail. |
| Symbolic shape analysis | Add a small local resolver | Resolve shapes from `Definition.get_input_shapes()`, `Definition.get_output_shapes()`, workload axes, FX `tensor_meta`, and simple AST shape expressions. Do not add `sympy`. |
| Operator derivation | Add local pattern/decomposition modules | Map reference structure into explicit `BoundGraphNode` families for attention, MoE, convolution, SSM/Mamba, embedding/positional, and linear projection. |
| Work formulas | Extend `amd_bound_estimates.py` | Implement per-family FLOP, read/write, intermediate, movement, confidence, and rationale formulas over `BoundGraph`, not raw source strings. |
| Artifact/reporting | Keep `amd_sol_v2.py`, `amd_score.py`, `scripts/run_dataset.py` | Continue emitting sidecar evidence and guarded AMD-native score reports; keep canonical trace JSONL unchanged. |

The arXiv 2603.19173 abstract frames SOLAR as a pipeline for analytically deriving hardware-grounded SOL bounds and SOL Score as a measure of closing the gap between a release-defined baseline and the hardware SOL bound. For this ROCm milestone, only that derivation concept should be mapped to AMD. Do not import the paper's NVIDIA Blackwell, BF16/FP8/NVFP4 hardware claims into ROCm score semantics.

## Recommended Stack Additions

### Local Modules

| Module | Purpose | Why |
|--------|---------|-----|
| `src/sol_execbench/core/scoring/amd_symbolic_shapes.py` | Workload-aware shape/dtype resolver for axis values, FX `tensor_meta`, AST literals, tuple/list expressions, `x.shape[i]`, `x.size(i)`, reshape/view arguments, and simple arithmetic. | Current extraction has concrete input/output shapes but no reusable resolver for derived intermediate dimensions. This is the highest-leverage addition. |
| `src/sol_execbench/core/scoring/amd_derivation_patterns.py` | Declarative pattern registry mapping FX/AST call names and operator sequences to `OpFamily`, attributes, and confidence. | Avoids turning `_CALL_CLASSIFIERS` into a long ad hoc table and makes family coverage auditable. |
| `src/sol_execbench/core/scoring/amd_operator_formulas.py` | Per-family formula helpers consumed by `amd_bound_estimates.py`, or a closely scoped internal section if the file stays small. | Keeps extraction separate from accounting. Formula outputs should remain `OperatorWorkEstimate`. |
| Existing `amd_bound_graph.py` | Extend, do not replace. | It already has `BoundTensor`, `BoundEdge`, `BoundGraphNode`, `OpFamily`, FX tracing, AST fallback, warnings, and JSON-safe serialization. |
| Existing `amd_bound_estimates.py` | Extend dispatch for currently unsupported families. | It already owns FLOP/byte/movement evidence and the unsupported degradation contract. |
| Existing `amd_sol_v2.py` | Preserve v2 sidecar schema if possible; add only backward-compatible fields through existing dictionaries. | Score reports already understand `scored`, `degraded`, and `unscored`. Schema churn is unnecessary unless coverage semantics cannot be expressed. |

### Python And PyTorch APIs

| API | Use | Guardrail |
|-----|-----|-----------|
| `ast.parse`, `ast.NodeVisitor` | Deterministic source fallback, call-chain extraction, return binding, assignment tracking, simple expression evaluation. | Use `visit_Constant`, not deprecated visitor names. Keep traversal order explicit for assignments and returns. |
| `torch.fx.symbolic_trace` | Capture common PyTorch reference functions without writing a parser for every tensor method. | FX does not support input-dependent dynamic control flow; failed tracing must keep deterministic `dynamic_trace_failed` fallback semantics. |
| `torch.fx.passes.shape_prop.ShapeProp` | Populate node tensor metadata from CPU sample inputs. | Keep CPU zero tensors and small metadata-only propagation; do not require ROCm hardware for derivation tests. |
| FX `node.meta["tensor_meta"]` | Get concrete intermediate shape/dtype evidence when available. | Treat missing metadata as inexact, not fatal. |
| `operator` module targets | Recognize binary arithmetic, matmul, comparisons, indexing-related operations. | Preserve unsupported nodes instead of dropping them. |

Do not adopt `torch.fx.experimental.symbolic_shapes`, `FakeTensorMode`, `torch.export`, ONNX, MLIR, Dynamo internals, or Triton parser infrastructure for v1.10. They may become useful later, but this milestone needs deterministic analysis of small `Definition.reference` functions with no new runtime dependency or compiler-stack commitment.

## Implementation Patterns

### 1. Extract, Then Classify, Then Estimate

Keep the pipeline explicit:

```text
Definition + Workload
  -> build_bound_graph()
  -> pattern/decomposition enrichment
  -> estimate_bound_work()
  -> build_amd_sol_bound_v2_artifact()
  -> score_amd_native_trace_workload()
```

`build_bound_graph()` should remain the single entry point. Add helper stages inside it rather than creating a separate public derivation API.

### 2. Use Family-Specific Decompositions

Represent high-level families as one family node with enough attributes to audit the formula, not as opaque taxonomy placeholders.

| Family | Extraction Pattern | Estimate Pattern |
|--------|--------------------|------------------|
| Attention | Q/K/V linear projections, `matmul(q, k^T)`, scale, softmax, dropout/no-op if inference, `matmul(prob, v)`, output projection. Recognize `scaled_dot_product_attention` as a direct family node. | Record QK GEMM, softmax passes, PV GEMM, projection GEMM, read/write/intermediate bytes, sequence/head dimensions, and mask/dropout caveats. |
| MoE | Router/top-k, expert gather/scatter, per-expert linear/MLP, combine weights. | Inexact unless active expert count and routing shape are statically known. Account routing movement and expert GEMM upper/lower bound evidence. |
| Convolution | `torch.nn.functional.conv*`, `torch.conv*`, module call names from FX, and AST `F.conv2d`. | Use N/C/H/W, output shape, kernel, stride, padding, dilation, groups. FLOPs = output elements * kernel volume * input channels per group * 2. |
| SSM/Mamba | `cumsum`, scan-like recurrences, selective scan function names, depthwise conv + projection patterns. | Usually inexact. Capture projection GEMMs, depthwise conv, scan pass count, state bytes, and unsupported warning for value-dependent recurrence if dimensions cannot be proven. |
| Embedding/positional | `embedding`, indexing/gather, arange/position id creation, rotary/sin/cos patterns. | Mostly memory movement plus elementwise rotation. Distinguish lookup bytes from generated positional tensors. |
| Linear projection | `linear`, `matmul + bias`, `@`, `einsum` forms equivalent to GEMM. | Supported when M/N/K are resolved; inexact when transpose/broadcast semantics are inferred but not proven. |

### 3. Make Shape Evidence First-Class

Add shape evidence to `BoundGraphNode.attributes` rather than changing node fields:

```python
{
    "trace_source": "torch.fx" | "ast",
    "shape_source": "fx_tensor_meta" | "definition_workload" | "ast_symbolic" | "missing",
    "symbolic_dims": {"B": 8, "S": 2048, "H": 4096},
    "formula_dims": {"M": 16384, "N": 4096, "K": 4096},
    "layout_notes": ["rhs_transposed"],
}
```

This keeps the v2 sidecar JSON-compatible and lets `amd_bound_estimates.py` consume richer evidence without a schema rewrite.

### 4. Preserve Degradation Semantics

Every recognized but partially modeled family should produce an `OperatorWorkEstimate` with `EstimateConfidence.INEXACT`, a non-empty rationale, and deterministic warnings. Truly unknown semantics should remain `EstimateConfidence.UNSUPPORTED`, making the aggregate `unscored` through the existing `amd_sol_v2.py` logic.

This is more paper-aligned than silently scoring partial graphs: SOLAR-style bounds are useful only when the evidence chain says what was modeled and what was not.

## Integration Points

| File | v1.10 Work |
|------|------------|
| `src/sol_execbench/core/scoring/amd_bound_graph.py` | Extend `_CALL_CLASSIFIERS`, FX node attributes, and `_AstBoundGraphExtractor` with pattern registry hooks, shape evidence, module/function aliases, and sequence recognition. |
| `src/sol_execbench/core/scoring/amd_bound_estimates.py` | Add dispatch handlers for `ATTENTION`, `MOE`, `CONVOLUTION`, `SSM_MAMBA`, and `EMBEDDING_POSITIONAL`; improve `LINEAR_PROJECTION` beyond generic GEMM where bias/broadcast/projection semantics are explicit. |
| `src/sol_execbench/core/scoring/amd_sol_v2.py` | Keep aggregate states. Consider adding coverage detail only via existing `warnings`, `coverage_summary`, and `operator_work_estimates` dictionaries unless a schema bump is unavoidable. |
| `src/sol_execbench/core/scoring/amd_score.py` | No formula change. Ensure degraded/unscored sidecars still block or warn exactly as today. |
| `scripts/run_dataset.py` | Keep current `--amd-score-report` / `--amd-sol-bound-dir` integration. Do not add dataset extraction workflows. Optionally add a derivation-only dry-run later if needed, but not as core stack. |
| Tests | Add focused CPU-only golden tests for graph extraction, shape evidence, per-family formulas, aggregate states, and dataset sidecar writing. |

## What Not To Add

| Do Not Add | Reason |
|------------|--------|
| New dataset/model extraction pipeline | Explicitly out of v1.10 scope. Use existing problem `definition.json` and `workload.jsonl`. |
| Real-hardware validation harness changes | Existing profiler/timing paths are enough; this milestone is derivation-only. |
| Hosted leaderboard or submission service | Out of scope and would blur AMD-native derived evidence with public leaderboard claims. |
| `sympy` | Simple integer shape expressions and workload axes are enough; symbolic algebra would add dependency and failure modes. |
| `networkx` | `BoundGraph` already provides the small evidence graph needed. |
| ONNX / MLIR / `torch.export` / Dynamo internals | Too much semantic and dependency surface for small embedded reference functions. |
| CUDA/NVIDIA compatibility layer | ROCm-only project; paper hardware claims must not be mapped onto AMD reports. |
| New canonical trace fields | Bound and score evidence belongs in sidecars and AMD-native reports. |
| Validated CDNA 3 / MI300X / CDNA 4 claims | Validation is explicitly deferred. Keep model status provisional unless evidence already exists. |

## Suggested Validation

```bash
uv run pytest \
  tests/sol_execbench/test_amd_bound_graph.py \
  tests/sol_execbench/test_amd_bound_estimates.py \
  tests/sol_execbench/test_amd_sol_v2.py \
  tests/sol_execbench/test_amd_native_score.py \
  tests/sol_execbench/test_run_dataset_amd_score.py \
  tests/sol_execbench/test_public_contract_guardrails.py

uv run --with ruff ruff check \
  src/sol_execbench/core/scoring \
  tests/sol_execbench/test_amd_bound_graph.py \
  tests/sol_execbench/test_amd_bound_estimates.py \
  tests/sol_execbench/test_amd_sol_v2.py
```

## Sources

- Repository: `.planning/PROJECT.md` - v1.10 SOLAR derivation scope and explicit deferrals.
- Repository: `src/sol_execbench/core/scoring/amd_bound_graph.py` - current BoundGraph IR, FX trace path, AST fallback, operator family enum, warnings.
- Repository: `src/sol_execbench/core/scoring/amd_bound_estimates.py` - current per-node FLOP/byte/movement estimate contract and unsupported fallback.
- Repository: `src/sol_execbench/core/scoring/amd_sol_v2.py` - sidecar schema, coverage summary, scored/degraded/unscored aggregate semantics.
- Repository: `src/sol_execbench/core/scoring/amd_score.py` - AMD-native score warnings, evidence refs, degraded/unscored handling.
- Repository: `scripts/run_dataset.py` - dataset report integration and sidecar writing path.
- Paper baseline: https://arxiv.org/abs/2603.19173 - SOL-ExecBench abstract, SOLAR-derived hardware SOL bounds, SOL Score, and NVIDIA Blackwell benchmark scope.
- Official docs: https://docs.pytorch.org/docs/stable/fx.html - FX symbolic tracing, IR, ShapeProp, and dynamic-control-flow limitations. Confidence: HIGH.
- Official docs: https://docs.python.org/3/library/ast.html - Python AST parsing and `NodeVisitor` traversal APIs. Confidence: HIGH.
- Official docs: https://docs.pytorch.org/docs/2.9/torch.compiler_fake_tensor.html - FakeTensor/dynamic-shape context; useful background but not recommended for this milestone. Confidence: MEDIUM because it is compiler-internals-oriented.
