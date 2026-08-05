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

## Current coverage boundary

The current focused readiness matrix is 41/41 on both paths. The last generated
repository-owned accounting report covered an earlier 32-workload subset, so it
is historical evidence rather than a result for the current denominator. This
document intentionally does not preserve that snapshot's mismatch counts; use
Git history when auditing it.

Until a new 41-workload report is generated, no repository-wide equality claim
may be made for model I/O, mandatory work, internal fusion accounting, limiting
resource, or formal bounds. The refresh is tracked in the
[active backlog](../../HANDSOFF.md).

The four remaining full-corpus Torchview failures are explicit backward
references in `instruction2triton/rmsnorm_bwd`. They remain fail-closed at graph
extraction because a forward-only Torchview trace cannot represent their
upstream-gradient dependency or gradient outputs.
