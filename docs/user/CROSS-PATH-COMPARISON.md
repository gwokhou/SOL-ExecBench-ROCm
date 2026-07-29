# Cross-Path SOLAR Comparison

SOLAR has two fixed extraction-to-IR paths:
`make_fx_aten` and `torchview_extended_einsum`. Successful replay on both paths
proves each graph against the same callable reference. It does not by itself
prove equal graph accounting or an equal formal lower bound.

The repository-owned comparison command keeps those claims separate:

```bash
uv run sol-execbench solar compare-paths \
  PATH/TO/MAKE_FX_ANALYSES \
  PATH/TO/TORCHVIEW_ANALYSES \
  --output path-comparison.json
```

The first root must contain `make_fx_aten` manifests and the second must contain
`torchview_extended_einsum` manifests. The comparator verifies every referenced
artifact hash and conversion-attestation subject before reading its contents.
It rejects mixed paths, malformed schemas, missing required artifacts, hash
drift, and failed attestations.

## Comparison dimensions

Every dual-ready workload reports five independent dimensions:

1. `external_reference_io` binds the analysis ID, reference digest,
   architecture, precision, verification policy and cases, execution identity,
   and the ordered live graph input/output signatures.
2. `model_io_accounting` compares aggregate model-I/O and deduplicated external
   fused-I/O elements and bytes.
3. `mandatory_resource_work` compares MAC accounting, precision splits,
   resource-model version, and per-resource mandatory work.
4. `fusion_intermediate_accounting` compares layer decomposition, unfused and
   intermediate traffic, audited fused/prefetched traffic, Orojenesis traffic,
   and orphan counts.
5. `formal_bound` compares resource seconds, limiting resource, bound
   components, bound kind, and final lower-bound seconds.

Differences use one of these reviewed causes:

- `extraction_topology_loss`
- `normalization_difference`
- `legitimate_dialect_decomposition_difference`
- `resource_model_bug`
- `formal_bound_policy_difference`

The report never selects a lower result, averages bounds, falls back to the
other path, or declares equal accounting from numerically matching replay.

Coverage is also fail-closed. Missing workloads are listed under
`missing_by_path`; unequal coverage makes the overall status `incomplete` and
the command exits nonzero even when all dual-ready workloads agree on
authoritative accounting.

## Focused dual-path result

The post-fix 41-workload focused matrix has 32 workloads ready on both paths.
For those 32:

| Dimension | Mismatches |
| --- | ---: |
| External reference/I/O identity | 0 |
| Graph-level model I/O | 0 |
| Mandatory resource work | 0 |
| Limiting resource and formal bound | 0 |
| Internal fusion/intermediate accounting | 32 |

The remaining internal differences are dialect decomposition differences:
ATen preserves exact granular operations, aliases, and views, while
extended-einsum normalizes or combines some of that topology. All 32 differ in
layer count, unfused traffic, and intermediate traffic; 21 differ in the number
of intermediate tensors and four differ in zero-work orphan counts. Audited
fused/prefetched traffic agrees, so these internal differences do not change
mandatory resource work, the limiting resource, or the final bound.

Three generic fixes established that result:

- graph producer topology, rather than missing tensor-role metadata, determines
  whether an input is internal;
- only layers reachable from declared graph outputs contribute work or I/O;
- a Torchview dtype-view preserves its explicit destination dtype instead of
  being normalized as a shape-only view.

The focused matrix still has nine Torchview coverage failures: four exact source
input binding failures and five graph extraction failures. They remain
fail-closed and keep the corpus comparison status `incomplete`; improving that
coverage is separate from explaining accounting for dual-ready workloads.

