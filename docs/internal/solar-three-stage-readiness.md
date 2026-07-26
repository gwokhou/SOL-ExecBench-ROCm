# SOLAR three-stage readiness assessment

> Internal assessment of the three stages from source-computation selection to
> executable SOLAR semantics. Snapshot: 2026-07-25. This is an engineering
> readiness report, not a release statement or an official SOL Score claim.

## 1. Executive summary

The three stages have materially different levels of completion:

| Stage | Estimated completion | Current status |
| --- | ---: | --- |
| Model/task to Problem | ~45% | Fixed seed authoring is reliable; automatic discovery and representative model coverage are incomplete |
| Problem to operator graph | ~75% | Forward tracing is mostly operational; backward integration and exact metadata remain incomplete |
| Operator graph to einsum graph | ~50-55% | The fail-closed proof boundary is strong, but only half of the current scored workloads pass exact replay |

The most important end-to-end result is:

| Population | Extraction passed | Strict conversion passed | Numerical verification passed |
| --- | ---: | ---: | ---: |
| 35 scored problems / 122 workloads | 118 / 122 (96.7%) | 84 / 122 (68.9%) | 61 / 122 (50.0%) |

Only 17 of the 35 scored problems pass extraction, strict conversion, and
verification for every workload. The implementation therefore has a mature
safety boundary but is not yet a closed formal-analysis pipeline for the
current corpus.

The completion percentages are expert estimates against the intended stage
goals, not line coverage or a mechanical average. Hard pass counts are reported
separately and should be preferred when tracking progress.

## 2. Stage definitions

This report uses the following responsibility split:

```text
source model or AKA task
        |
        | semantic boundary selection, standalone reference construction,
        | workload derivation, characterization, deduplication, sampling
        v
Problem: definition + reference + workloads
        |
        | concrete execution tracing and exact input/output observation
        v
operator graph
        |
        | strict operation conversion, semantic annotation, executable replay
        v
einsum graph + conversion attestation
```

The first stage optimizes for representative coverage of a target workload
distribution. The second stage optimizes for a faithful graph of one concrete
execution path. The third stage optimizes for exact executable semantics and
must fail closed whenever equivalence cannot be proven.

## 3. Audit scope and environment

The assessment was performed on:

| Item | Value |
| --- | --- |
| Repository revision | `5aedb82a6233b88c958cec8a724819dcb04e0a7a` |
| PyTorch | `2.11.0+rocm7.2` |
| HIP runtime reported by PyTorch | `7.2.26015` |
| Device | AMD Radeon RX 9060 XT |
| Architecture | `gfx1200` |

Three forms of evidence were used:

1. Static inspection of the authoring, extraction, conversion, semantics, and
   verification boundaries.
2. The repository's relevant test suites:

   ```bash
   uv run pytest -n 0 tests/solar tests/sol_execbench/core/dataset
   ```

   Result: `442 passed`.
3. Host ROCm execution over the committed corpus:
   - reference sanity and AKA-oracle checks over all 37 problems and all 128
     workloads;
   - extraction, strict conversion, and the standard three-seed by three-pattern
     replay protocol over all 35 scored problems and 122 scored workloads.

The corpus-wide three-stage audit used the same production functions as the
public bridge:

- `build_input_factory`;
- `extract_operator_graph`;
- `convert_operator_graph`;
- `verify_callable_conversion`.

It intentionally stopped before SOL bound analysis because this report assesses
only the three stages above.

## 4. Stage 1: model/task to Problem

### 4.1 What is implemented

The current AKA-derived corpus has a useful authoring and integrity foundation:

- committed `definition.json`, `reference.py`, and `workload.jsonl` artifacts;
- Definition and Workload schema validation;
- pinned AKA revision and per-task checksums;
- source provenance and entry roles;
- target-aware compatibility selection;
- atomic materialization and post-materialization audit;
- dtype-based tolerance construction;
- reference sanity and optional cross-checking against AKA's `module_fn`.

The corpus implementation is owned by
[`aka_corpus.py`](../../src/sol_execbench/core/dataset/aka_corpus.py), while the
offline authored seed is defined in
[`aka_author_seed.py`](../../scripts/internal/aka_author_seed.py).

The host audit found:

- 37/37 problems passed standalone reference sanity;
- all 128 workloads passed the sanity check;
- 28/37 problems were directly cross-checked against AKA `module_fn`;
- 9/37 were skipped because the signature was restructured or no resolvable
  torch2* oracle was available;
- no executed AKA cross-check failed.

### 4.2 Why this is not yet model-to-Problem extraction

The current authoring script contains a hard-coded `SPECS` list. Its references,
axes, dtypes, workload values, classifications, and descriptions are curated
per problem. It does not discover subgraphs from a model or automatically lift
an arbitrary AKA candidate.

A deterministic selector exists in
[`aka_selector.py`](../../src/sol_execbench/core/dataset/aka_selector.py), but it
has no production caller or focused test and is not used by the authoring
script. After satisfying minimum combinations, it fills remaining slots in
sorted order rather than optimizing a multi-axis coverage objective.

The manifest's `formal_coverage_requirements.axes` is generated by counting the
already selected specs. This proves that the manifest truthfully describes the
selection; it does not establish an independent target distribution or prove
that the selection is representative.

The current source is also explicitly `ecosystem_grounded`. It lacks the model
identity, model domain, source subgraph location, forward/backward role
distribution, compute-intensity characterization, and shape-distribution
evidence needed for a model-grounded representativeness claim.

### 4.3 Validation limitations

[`aka_equivalence_check.py`](../../scripts/aka_equivalence_check.py) is valuable
but not a complete equivalence proof:

- its default checks two workloads per problem;
- restructured signatures are reported as skipped rather than adapted and
  executed;
- instruction2triton problems do not receive an original-oracle cross-check;
- dictionary and tuple outputs are reduced to their first tensor for sanity;
- expected output metadata is taken from the first declared output.

The last two limitations are especially important for backward problems with
multiple gradients.

### 4.4 Assessment

This stage is best described as a reliable, manually curated seed builder. It
is substantially complete for regenerating and auditing the current fixed
corpus, but incomplete against the intended goal of broad, reproducible
model-to-Problem discovery and stratified selection.

Estimated completion: **approximately 45%**.

## 5. Stage 2: Problem to operator graph

### 5.1 What is implemented

The public extraction boundary in
[`extraction.py`](../../src/solar/graph/extraction.py) records:

- the exact tensor source arguments observed by the reference;
- which source argument indices were used;
- source input shapes and dtypes;
- reference output arity, shapes, and dtypes;
- the torchview operator graph.

[`dispatch_coverage.py`](../../src/solar/graph/dispatch_coverage.py) rejects
fully or partially untracked tensor dispatches when torchview loses
`RecorderTensor` lineage. This prevents a lower bound from being computed from
the visible remainder of a partial graph.

The torchview processor also preserves argument ordering, connections,
operation parameters, shapes, and dtypes. Mixed-precision graphs reject an
unknown intermediate dtype rather than silently assigning FP32.

### 5.2 Corpus results

Of the 122 scored workloads:

- 118 produced an operator graph;
- all workloads in 34 of the 35 scored problems produced an operator graph;
- all four `rmsnorm_bwd` workloads failed extraction.

The 96.7% artifact success rate is strong, but it is not by itself a 96.7%
accuracy claim. Some graphs that were emitted exposed missing input or
parameter metadata only when strict conversion attempted to consume them.
Those seam failures affect 30 workloads dominated by functional
linear/layernorm graphs.

### 5.3 Remaining limitations

The public path traces one concrete seed and therefore one concrete execution
path. Later replay uses other seeds and input patterns, but it does not retrace
shape-dependent or data-dependent branches. Corpus workload design must
therefore supply path coverage, or extraction must publish and validate
multiple trace variants.

A pinned AOTAutograd implementation exists in
[`backward_processor.py`](../../src/solar/graph/backward_processor.py), including
gradient replay validation, alias information, and mutation effects. It is not
connected to `solar.api`, `extract_operator_graph`, or the benchmark bridge.
The only production references to `BackwardProcessor` are inside its own
module. The current backward problem consequently goes through ordinary
torchview tracing and fails.

The visual-graph fallback and several torchview metadata repairs are useful for
compatibility, but they increase the need for corpus-wide conversion replay as
the final authority.

### 5.4 Assessment

Forward extraction for tensor programs without complex control flow is largely
operational and safely fails on lost lineage. Exact parameter metadata,
multi-path traces, and an integrated backward route remain incomplete.

Estimated completion: **approximately 75%**.

## 6. Stage 3: operator graph to einsum graph

### 6.1 What is implemented

The formal conversion boundary in
[`conversion.py`](../../src/solar/einsum/conversion.py) uses strict mode and:

- accepts only reviewed expansion handlers;
- requires exact source-input binding;
- rejects missing or ambiguous bindings;
- preserves exact reference output arity and metadata;
- emits current-schema executable semantic graphs.

[`semantics.py`](../../src/solar/einsum/semantics.py) requires each operation to
carry explicit input/output names, shapes, dtypes, ordered arguments, keyword
arguments, mutation effects, alias effects, and operation-specific required
parameters.

[`verify_callable_conversion`](../../src/solar/verification/einsum.py) executes
the converted graph over three seeds and the `random`, `zeros`, and `boundary`
patterns. It checks:

- output arity;
- shape and dtype;
- numerical tolerance and maximum error policy;
- integer and Boolean equality;
- non-finite values;
- input mutation;
- output/input alias relationships.

This is the strongest part of the implementation. Failed conversions do not
silently become approximate formal bounds.

### 6.2 Corpus results

Of the 122 scored workloads:

- 84 produced a strict einsum graph;
- 61 passed the complete replay protocol;
- 17 of 35 scored problems passed for every workload.

The 61 workloads that did not reach a verified graph failed at their first
blocking stage as follows:

| Failure class | Workloads | Representative symptoms |
| --- | ---: | --- |
| Backward graph extraction | 4 | `rmsnorm_bwd` cannot be traced by the public path |
| Input/parameter binding and metadata | 30 | layernorm start-count mismatch; functional linear lacks explicit input metadata |
| Unsupported exact operation | 4 | `vstack` in the feedforward problem |
| Exact ATen argument preservation/replay | 18 | extra `max_pool2d` argument, missing `instance_norm.cudnn_enabled`, leaked `_stacklevel`, invalid slice argument |
| Numerical equivalence | 5 | four BF16 SiLU-and-mul workloads and one matrix-vector workload |
| **Total** | **61** | |

The 84-to-61 drop is evidence that the numerical verifier is doing meaningful
work: 23 structurally valid graphs were rejected because they could not be
executed with exact reference semantics.

### 6.3 Test gap

The relevant test selection reports `442 passed`, and the unit tests cover
handlers, graph repairs, strict semantics, verification artifacts, aliasing,
backward serialization, and representative pipeline cases.

However,
[`test_pipeline_integration.py`](../../tests/solar/test_pipeline_integration.py)
uses a small synthetic case list. No test enumerates the committed
`problems/AMD_AKA` manifest and requires every scored workload to pass
extraction, conversion, and verification. This explains how the unit suite can
remain green while only half of the scored workloads have a complete formal
conversion.

### 6.4 Assessment

The conversion architecture and rejection policy are mature. Operation and
argument coverage for the actual corpus are not. Safety maturity should not be
confused with workload completion.

Estimated completion: **approximately 50-55%**.

## 7. Recommended priority order

### P0: make corpus readiness observable

Add a maintained corpus-stage audit command and a CI/self-hosted ROCm job that
publishes a matrix containing, for every scored workload:

- extraction status and stable reason code;
- strict conversion status and stable reason code;
- verification status and attestation digest;
- trace, graph, reference, architecture, and workload identities.

Formal corpus readiness must require every scored workload to pass. Unsupported
workloads must remain visible rather than being silently removed.

### P0: close the current corpus seam

Fix the highest-impact failure classes in this order:

1. Functional linear and layernorm source-input/parameter binding: 30 workloads.
2. Exact ATen argument normalization and replay: 18 workloads.
3. Backward routing through the AOTAutograd processor: 4 workloads.
4. Exact `vstack` semantics: 4 workloads.
5. The five numerical-equivalence failures, without weakening tolerances to
   accept unexplained error.

### P1: strengthen Problem construction

- Replace the hard-coded-only workflow with a candidate intake and
  characterization pipeline.
- Connect and test deterministic stratified selection.
- Define coverage targets independently of the selected set.
- Add source model, model domain, source subgraph, precision mode, shape regime,
  compute intensity, and forward/backward metadata.
- Add semantic deduplication and near-duplicate reporting.
- Adapt restructured AKA signatures and compare every declared output over every
  admitted workload.

### P1: cover execution paths

- Record whether a Problem is single-path under all workloads.
- Retrace distinct shape- or data-dependent paths when necessary.
- Bind each accepted trace variant to the workloads for which it is valid.
- Reject a Problem from formal readiness if its workload paths are not covered.

## 8. Completion criteria

The following criteria make the stage goals measurable.

### Stage 1 complete for a release corpus

- every selected problem has deterministic source and workload provenance;
- target coverage axes and quotas are defined before selection;
- selection is reproducible from a pinned candidate pool;
- every semantic rewrite is independently cross-checked;
- every declared output and every admitted workload is validated;
- model-grounded and ecosystem-grounded claims remain separate.

### Stage 2 complete for a release corpus

- every scored workload produces a non-partial operator graph;
- every source input and reference output has an exact binding;
- forward and backward problems use supported, verified extraction paths;
- distinct control-flow paths are traced or explicitly ruled out;
- downstream metadata validation finds no extraction-seam defects.

### Stage 3 complete for a release corpus

- every scored workload converts in strict mode;
- every converted graph passes the standard multi-seed, multi-pattern replay;
- all mutation and alias behavior matches the reference;
- corpus-wide status is enforced by CI and content-addressed release evidence.

For the current manifest, the immediate numerical target is therefore:

```text
122 / 122 extraction passed
122 / 122 strict conversion passed
122 / 122 conversion verification passed
35 / 35 scored problems fully verified
```

These denominators must be derived from the manifest rather than hard-coded so
that corpus expansion cannot weaken the gate.

## 9. Final conclusion

The repository already has a strong formal philosophy: incomplete traces,
unknown semantics, ambiguous bindings, and numerical disagreement are rejected
instead of approximated. That is the correct foundation.

The remaining work is primarily closure and coverage:

- Stage 1 must evolve from manual ecosystem seed authoring into reproducible,
  independently targeted corpus construction.
- Stage 2 must integrate backward extraction and eliminate metadata defects at
  the tracing boundary.
- Stage 3 must turn its strong verifier into a corpus-wide admission gate and
  bring the current 50% verified-workload rate to 100% before formal release.

