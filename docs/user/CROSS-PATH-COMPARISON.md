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

The current focused readiness matrix is 41/41 on both paths across eight scored
problems. Validate that exact current-manifest denominator without GPU work:

```bash
uv run python scripts/internal/solar/run_cross_path_focus.py --check
```

On a publishing host, run the resumable 82-analysis route and produce the
comparison only after every workload has a formal publication on both paths:

```bash
uv run python scripts/internal/solar/run_cross_path_focus.py \
  --output data/outputs/solar-cross-path-focus \
  --orojenesis-home /path/to/orojenesis --device cuda:0 --resume
```

The runner validates that all eight entries are scored and retain their exact
41-workload denominator, fixes path order, isolates output by IR path, and
fails closed before comparison on any non-formal worker result. Existing
per-workload directories are skipped only with `--resume`; the final comparator
then validates their schemas and content hashes.

The current ignored report at
`data/outputs/solar-cross-path-focus-cycle2-c84869e/path-comparison.json` covers
all 82 path analyses for the 41 unique workloads and has SHA-256
`7329cf39ad86937fd19a88a9b9ee39c9597ebf33ff244a2d660588c341765b60`.
Twenty-seven workloads have status `matched_with_dialect_differences`;
fourteen have status `different` because of normalization differences. All 41
record legitimate dialect/decomposition differences, and the fourteen
normalization differences remain visible rather than being coerced to equality.

The report has no external-reference-I/O mismatch and review found no remaining
resource-model-bug classification. Raw model-I/O, mandatory-work,
fusion/intermediate, limiting-resource, and formal-bound fields may still differ
as consequences of the recorded dialect, decomposition, or normalization
boundary. The report is content-addressed local evidence under the ignored
`data/outputs/` tree, not publisher release authority.

The four remaining full-corpus Torchview failures are explicit backward
references in `instruction2triton/rmsnorm_bwd`. They remain fail-closed at graph
extraction because a forward-only Torchview trace cannot represent their
upstream-gradient dependency or gradient outputs.
