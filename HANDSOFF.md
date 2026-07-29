# SOLAR Dual-Path Handoff

## Repository state

Handoff date: 2026-07-29

The dual-path implementation baseline is recorded by these DCO commits:

```text
35e0f836 Add fail-closed SOLAR path comparison
24dfe9d6 Enable fixed dual-path SOLAR analysis
c5510916 Expand AKA workload and correctness contracts
```

The commits contain `Signed-off-by: Guohao Zhang
<akidezhang@outlook.com>`.

## What is implemented

SOLAR exposes exactly two reviewed extraction-to-IR paths:

```text
torchview_extended_einsum  (default)
make_fx_aten
```

The `IRPath` enum binds the extractor, IR kind, and canonical graph filename.
CLI and JSON/worker IPC boundaries normalize strings once. Internal requests
hold a typed `IRPath`; properties and downstream workers do not repeatedly
normalize it.

The following behavior is enforced:

- `analyze`, `corpus-audit`, and `release-build` select one path with
  `--backend`.
- Extractor and IR cannot be selected independently.
- A command never falls back to the other path.
- One release root cannot mix paths or resume with a different path.
- Extended-einsum writes `einsum_graph.yaml`; ATen writes `aten_graph.yaml`.
- Requests, results, attestations, corpus records, manifests, and release
  indexes bind `ir_path`.
- Resume and release verification reject path or artifact-name drift.
- Exact ATen target/overload, output slots, aliases, and mutation effects are
  preserved and replayed.
- Structured inputs such as `cu_seqlens` are protected during zeros/boundary
  verification through `preserved_input_indices`.
- Resource analysis uses `amd_resource_v2`.
- Unknown exact operations, unclassified resources, ambiguous Torchview input
  binding, incomplete topology, and incomplete Orojenesis evidence remain
  fail-closed.

The default path has deliberately not changed from
`torchview_extended_einsum`. New workloads may opt into `make_fx_aten`
explicitly.

## Verification already completed

### Repository gates

The final tree passed:

```bash
uv run pytest tests/
uv run --no-sync ty check
uv run --no-sync python scripts/check_coupling.py
uv run --no-sync python scripts/check_readability.py
uv run --no-sync python scripts/check_production_reachability.py
uv run --no-sync python scripts/check_current_docs.py
uv run --no-sync python scripts/check_schema_versions.py
uv run --with ruff ruff check .
uv run --with ruff ruff format --check .
git diff --check
```

Do not raise readability, coupling, or quality baselines to accommodate later
changes.

### GPU audits

The host used for validation was:

```text
AMD Radeon RX 9060 XT
gfx1200
ROCm device cuda:0
```

Current focused conversion/replay results for the eight new problems and 41
workloads:

| Path | Conversion/replay |
| --- | ---: |
| `make_fx_aten` | 41/41 ready |
| `torchview_extended_einsum` | 41/41 ready |

The previous nine Torchview failures were fixed generically:

- 0-D RecorderTensor edges now retain their exact producer identity;
- positional and keyword tensor arguments share one ordered trace contract;
- root TensorNodes carry their exact reference source indices.

No problem-name special cases or shape/dtype source-binding guesses were added.
Strict resource analysis and cross-path accounting have not been rerun for the
nine newly ready workloads, so 41-workload accounting equivalence is not
established.

The current full scored-corpus conversion audit produces:

| Path | Ready | Failure summary |
| --- | ---: | --- |
| `make_fx_aten` | 163/163 | none |
| `torchview_extended_einsum` | 159/163 | 4 graph extraction |

The four remaining failures are every workload in
`instruction2triton/rmsnorm_bwd`. Vendored Torchview executes its trace under
`torch.no_grad()` and cannot represent the reference's explicit autograd
backward graph. Enabling gradients only records the forward graph and omits
`grad_output` plus both gradient outputs, so that is not a valid coverage fix.
These workloads remain fail-closed at `graph_extraction`.

The current 159/163 result is development evidence. Rerun the complete matrix
on the exact reviewed source before publishing evidence.

Temporary evidence currently exists at:

```text
/tmp/solar-audit-makefx-new41-final/focused-summary.json
/tmp/solar-analysis-makefx-new41-final/analysis-summary.json
/tmp/solar-audit-torchview-new41-final/focused-summary.json
/tmp/solar-analysis-torchview-new41-final/analysis-summary.json
/tmp/solar-dual-path-comparison-final.json
/tmp/solar-corpus-audit-makefx-full-v1/summary.json
/tmp/solar-corpus-audit-torchview-coverage-v6/summary.json
```

These files are not committed and may disappear. They are development evidence,
not publication artifacts.

### Cross-path comparison

Thirty-two new workloads are ready on both paths. Their exact replay succeeds
against the same reference contract, but the current analysis artifacts do not
establish cross-path equivalence:

```text
dual ready workloads:          32
model I/O accounting mismatch: 32
resource-work mismatch:        12
formal-bound mismatch:         32
```

The comparison script used for this check is `/tmp/compare_solar_paths.py`.
It is an ad hoc diagnostic, not a repository-owned verifier.

Never select the lower or otherwise more favorable result automatically. A
successful replay on both paths proves each path against the reference; it does
not prove that their graph accounting or formal lower bounds are equivalent.

## Next work

### P0: Produce and review release evidence

Official scoring is intentionally still unavailable:

```yaml
official_scoring:
  status: unavailable
  baseline_id: rx9060xt-gfx1200-reference-v2
  reason_code: baseline_v2_release_evidence_pending
```

Do not change this policy merely because conversion readiness is complete.
Availability requires a reviewed, content-addressed release bundle.

1. Start from a clean checkout of `24dfe9d6` plus any reviewed follow-up
   commits.
2. Reproduce the pinned Orojenesis mapper:

   ```bash
   scripts/internal/orojenesis/verify_reproducible_build.sh \
     out/orojenesis-reproducible
   ```

3. Verify that the two clean builds, provenance, and mapper digest are
   byte-identical. The reviewed digest currently allowlisted in
   `src/solar/analysis/orojenesis/configuration.py` is:

   ```text
   18591892b1ecec3264ec729b0e457ec9f22422993f656ece40dba809c032d77a
   ```

   Do not trust a locally self-declared provenance file or silently update the
   allowlist.

4. Rerun the complete scored-corpus audit on the exact release source:

   ```bash
   uv run sol-execbench solar corpus-audit \
     /tmp/solar-corpus-audit-makefx-release \
     --backend make_fx_aten \
     --device cuda:0

   uv run sol-execbench solar corpus-audit \
     /tmp/solar-corpus-audit-torchview-release \
     --backend torchview_extended_einsum \
     --device cuda:0
   ```

5. Build the canonical baseline and execute it in the pinned container:

   ```bash
   uv run sol-execbench baseline release-build out/release-makefx \
     --baseline-id rx9060xt-gfx1200-reference-v2 \
     --source-revision SOURCE_GIT_SHA

   ./scripts/run_docker.sh -- sol-execbench baseline release-run \
     /outputs/release-makefx/baseline/plan.json
   ```

6. Build the formal SOLAR denominator with one explicit path. MakeFX is the
   current practical candidate because all 163 workloads pass its strict
   conversion audit:

   ```bash
   uv run sol-execbench solar release-build out/release-makefx \
     --backend make_fx_aten \
     --orojenesis-home /path/to/reviewed/orojenesis
   ```

   If a Torchview release is also required, use a separate release root. Never
   mix the two paths within one index.

7. Build statements, assemble the bundle, and run the official verifier as
   documented in `docs/user/RELEASE-SCORING.md`.
8. Review every missing/unsupported contraction proof and every release
   verifier failure. Do not downgrade a formal requirement to diagnostic
   output.
9. Only after the repository-owned release evidence is complete and reviewed
   should a separate change update the official-scoring policy.

No hardware peak recalibration is required solely for `amd_resource_v2`: the
resource kinds and peak values did not change. The architecture profile hash
is pinned to:

```text
55cd3f60ead976732130ab23c9e76b526f9435e2fa7e100707b1c75ae1a459cb
```

If peak values or resource kinds change later, treat that as a new calibration
task rather than reusing this conclusion.

### P1: Explain cross-path accounting differences

Before claiming equivalence, add a repository-owned comparison contract that
separates:

- external reference input/output identity;
- graph-level model I/O accounting;
- per-resource mandatory work;
- fusion and intermediate accounting;
- limiting resource and final lower bound.

For each mismatch, identify whether it is:

- an extraction/topology loss;
- a normalization difference;
- a legitimate dialect decomposition difference;
- a resource-model bug;
- or a formal-bound policy difference.

The comparison must report differences and fail closed. It must not choose a
preferred path, average bounds, or treat numerically matching replay outputs as
proof of equal resource work.

### P2: Improve Torchview coverage generically

The generic source-binding, tensor-kwargs, functional-linear, multi-output
reduction, and non-finite scalar gaps are closed. The complete matrix is
159/163. The remaining work is explicit backward-reference support:

1. Define a reviewed Torchview-path contract that represents backward
   operations, upstream gradients, and ordered gradient outputs explicitly.
2. Do not reuse the MakeFX extractor or restore a prohibited cross-path
   extractor/IR combination.
3. Add focused CPU exact-replay tests for explicit backward references.
4. Rerun all four `rmsnorm_bwd` GPU rows, then the complete 163-workload matrix.
5. Keep the current graph-extraction failures until the full dependency graph
   is represented and exact replay succeeds.

Do not count a forward-only trace as improvement: it omits the defining
backward work and is not equivalent to the reference.

## Important invariants

- Keep `torchview_extended_einsum` as the default unless a separate reviewed
  contract change explicitly switches it.
- Do not restore `--extractor`.
- Do not restore
  `src/solar/ir/extended_einsum/make_fx_conversion.py`; that module represented
  a prohibited cross-path combination.
- Do not add automatic fallback, per-workload backend selection, or resume
  across paths.
- Keep exact target/overload and explicit alias/mutation effects.
- Do not replace exact replay with an operation-name whitelist.
- Keep structured input preservation in the request, attestation, manifest,
  and resume identity.
- Reachable or effectful uninitialized allocations must remain rejected.
- Unknown operations and unclassified resource work must remain errors.
- Do not commit `/tmp`, downloaded data, benchmark output, kernels, tokens, or
  proprietary evidence.
- GPU conclusions require host execution when the sandbox hides `/dev/kfd` or
  `/dev/dri`.
- Run commits with DCO signing:

  ```bash
  git commit -s -m "Imperative summary"
  ```

## Key implementation locations

```text
src/solar/ir/contracts.py
    IRPath and fixed extractor/IR/artifact bindings

src/solar/contracts.py
    typed conversion request and manifest contract

src/sol_execbench/core/solar_bridge/
    CLI/process boundary, corpus audit, release path identity

src/solar/graph/reference_serializer.py
src/solar/ir/aten/conversion.py
src/solar/verification/
    exact ATen schema/effects/replay and protected inputs

src/solar/graph/torchview/
src/solar/ir/extended_einsum/torchview/
    exact source binding and multi-output Torchview topology

src/solar/analysis/resources.py
src/solar/rocm/profiles/RX_9060_XT.yaml
    amd_resource_v2 formulas and pinned architecture identity

docs/user/RELEASE-SCORING.md
    authoritative release workflow
```

Before changing code, re-read `AGENTS.md` and `/home/guohao/.codex/RTK.md`.
Repository shell commands must be prefixed with `rtk`.
