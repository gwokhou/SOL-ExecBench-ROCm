# v1.11 Release Closure

v1.11 adds a local paper-dataset parity inventory and ROCm execution-closure
workflow. It improves auditability of the public SOL ExecBench dataset surface,
but it is not full 235-problem ROCm validation, not original 124-model
extraction parity, not upstream SOLAR parity,
not NVIDIA B200 or Blackwell equivalence, and not hosted leaderboard readiness.

## Artifact Matrix

| Artifact | What It Proves | What It Does Not Prove |
|----------|----------------|-------------------------|
| Dataset manifest | Local acquisition/layout can be inspected with category counts and checksums. | ROCm readiness, execution success, paper-level validation, hosted leaderboard parity, or upstream SOLAR equivalence. |
| Inventory | Canonical `Definition` and `Workload` files can be parsed or diagnosed with deterministic denominators. | That every workload can run on ROCm or that paper-scale extraction has been reproduced. |
| Readiness | Static blockers and ready-to-attempt workloads are visible with reason codes and next actions. | Execution success, scoring success, full validation, or hardware validation. |
| Ready subset | Ready workload refs can be selected without mutating canonical dataset files. | Complete dataset coverage or paper parity. |
| Execution closure | Bounded ready-subset runs can be joined to traces, failures, skips, missing traces, and evidence gaps. | Full 235-problem validation, hosted leaderboard readiness, or paper-level results. |
| Parity gap report | Manifest, inventory, readiness, execution closure, and derived evidence can be summarized by denominator, blocker, and evidence family. | A validation certificate or leaderboard result. |
| AMD-native score report | Local ROCm-derived score interpretation can be audited through trace, baseline, AMD SOL, hardware-model, and SOLAR derivation refs. | NVIDIA B200 equivalence, upstream SOLAR equivalence, or new real-hardware validation. |

## Deferred Validation

The following remain future work unless a later artifact explicitly records the
required evidence:

- full public 235-problem real-hardware validation,
- original 124-model / 7,400-subgraph extraction and curation reproduction,
- upstream NVlabs/SOLAR equivalence comparison,
- MI300X / CDNA 3 full-suite validation,
- CDNA 4 validation,
- NVFP4 and MXFP4 validation,
- hosted leaderboard or submission service.

## Public Contract Boundary

v1.11 sidecars do not modify canonical `definition.json`, `workload.jsonl`,
`solution.json`, trace JSON, primary `sol-execbench` CLI behavior, AMD SOL v2
sidecars, or SOLAR derivation sidecars. New report and closure fields remain
sidecar-only.
