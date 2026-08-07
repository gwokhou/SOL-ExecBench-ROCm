# Reviewed retirement plan — superseded diagnostic roots (2026-08-07)

This is the reviewed garbage-collection plan that HANDSOFF.md requires before
any deletion of the resolved legacy/debug/cache roots. It provides the exact
dry-run inventory, byte totals, reachability proof, cold-archive decision, and
the explicit approval checklist. **No data has been deleted by producing this
plan.** Deletion is executed only after explicit approval of the resolved
targets below.

The inventory is reproducible on demand through the lifecycle toolchain:

```bash
sol-execbench --format json diagnostics lifecycle retirement-plan
# or the thin script wrapper
uv run python scripts/plan_diagnostic_retirement.py
```

The planner lives in the lifecycle package
(`sol_execbench.core.bench.performance_model.lifecycle.retirement`), is
audit-only, never deletes or moves data, and only measures and proves
reachability.

## Scope

Resolved targets (from the HANDSOFF retirement audit and the Cycle 3
boundary), excluding the documented current release `p0-release-36e44fb/`:

| Target | Bytes | Files | Registry-reachable | Decision |
|---|---|---|---|---|
| `data/outputs/microarchitecture-diagnostics-v3` | 8.41 GB | 32,536 | no | cold-archive source evidence, then reclaim |
| `data/outputs/microarchitecture-diagnostics-v6` | 6.93 GB | 33,273 | no | cold-archive source evidence, then reclaim |
| `data/outputs/orojenesis-reproducible-9d17c17` | 1.51 GB | 449 | no | delete (reproducible) |
| `data/calibration` | 0.00 GB | 3 | no | delete (marked NON_CANONICAL) |
| `data/local-evidence` | 0.00 GB | 9 | no | delete (marked NON_CANONICAL) |
| `data/outputs/p0-release-0db2c5e` | 312.6 MB | 1,176 | no | delete (superseded attempt) |
| `data/outputs/p0-release-5aedb82` | 0.7 MB | 254 | no | delete (superseded attempt) |
| `data/outputs/p0-release-72612c3` | 297.2 MB | 999 | no | delete (superseded attempt) |
| `data/outputs/p0-release-82be9c8` | 105.8 MB | 656 | no | delete (superseded attempt) |
| **Total** | **17.57 GB** | 69,355 | **0 reachable** | |

## Reachability proof

The lifecycle registry under `data/store/` contains exactly one object: a
`DiagnosticDesignManifest` whose `policy_hashes.root` points at a `tmp_path`
pytest fixture, unrelated to any candidate. The content-addressed blob store
is empty (0 blobs). No lifecycle manifest, run-state object, or typed receipt
references any of the nine targets by path. The v3/v6 corpus files are the
archival `diagnostic_validation_corpus.v6` form, which the current v7
toolchain deliberately cannot read; they are not cited by any current
registry object. Therefore **all nine targets are unreachable by the
registry**, satisfying the reachability gate before deletion.

## Cold-archive decision

- **`microarchitecture-diagnostics-v3` / `microarchitecture-diagnostics-v6`**
  are superseded collection generations whose per-case SOLAR manifests and
  evidence manifests are source-audit material. Before reclaiming the root,
  retain a curated copy of the source evidence (the corpus declarations and
  per-case manifests, tens of MB) in cold storage; the bulk (Orojenesis
  search output such as `timeloop-mapper.oaves.csv`, smoke, pilot, and
  debug-staging trees) is reproducible and can be reclaimed. No ROCPD
  databases were found in either root.
- **`orojenesis-reproducible-9d17c17`** is unreferenced Orojenesis build
  output; it is reproducible and safe to reclaim directly.
- **`data/calibration`** and **`data/local-evidence`** are already marked
  `NON_CANONICAL.md` and have no production consumers.
- **`p0-release-*` (excluding `p0-release-36e44fb`)** are superseded release
  attempts predating the governed packager.

## Approval checklist

An approver must confirm all of the following before the deletion commands
run:

- [ ] The dry-run inventory above matches the current `plan_diagnostic_retirement.py`
      output and the reported byte totals.
- [ ] Reachability holds (planner reports `reachable_targets: 0`).
- [ ] A cold-archive copy of the v3/v6 source evidence exists at the chosen
      cold location before those roots are reclaimed.
- [ ] The documented current release `p0-release-36e44fb` and the
      `microarchitecture-diagnostics-v7*` roots are **not** in scope.
- [ ] Explicit approval is recorded for each resolved target.

## Approval-gated execution (after the checklist is satisfied)

Cold-archive the v3/v6 source evidence first (approver-selected location):

```bash
mkdir -p /path/to/cold/archive
tar --sort=name -C data/outputs -cf - \
  microarchitecture-diagnostics-v3/corpus \
  microarchitecture-diagnostics-v6/preregistered-corpus/cases \
  | tar -C /path/to/cold/archive -xf -
```

Then reclaim the approved targets:

```bash
rm -rf \
  data/outputs/microarchitecture-diagnostics-v3 \
  data/outputs/microarchitecture-diagnostics-v6 \
  data/outputs/orojenesis-reproducible-9d17c17 \
  data/calibration \
  data/local-evidence \
  data/outputs/p0-release-0db2c5e \
  data/outputs/p0-release-5aedb82 \
  data/outputs/p0-release-72612c3 \
  data/outputs/p0-release-82be9c8
```

`data/outputs/` is git-ignored, so none of these paths are tracked. Re-run
`plan_diagnostic_retirement.py` after reclamation to confirm the targets are
gone and no registry reference was lost.
