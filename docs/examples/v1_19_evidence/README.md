# v1.19 Evidence Demo Fixtures

These files are demo-only fixtures for the v1.19 evidence guide. They show
small JSON and Markdown report shapes for researchers who want to inspect the
artifact layout before generating local evidence.

Start with the central guide:
[v1.19 evidence guide](../../internal/v1_19_evidence_guide.md).

The fixtures are diagnostic-only and sidecars/reports only. They do not change
canonical Trace, Definition, Workload, Solution contracts, correctness, timing,
score, or evaluator semantics.

Required boundaries:

- no full 235-problem paper validation
- no upstream SOLAR parity
- no score authority
- no leaderboard readiness
- no CDNA3-family validation, including MI300X, and no CDNA4 validation
- no native-host ROCm Matrix validation
- no new-hardware validation

## Files

| File | Purpose |
| --- | --- |
| execution_closure.demo.json | Demo execution closure sidecar with relative trace and log refs. |
| paper_denominator.demo.json | Demo paper denominator report with source refs and false authority fields. |
| paper_denominator.demo.md | Demo Markdown rendering for denominator interpretation. |
| matrix_schema_export.demo.json | Compact Matrix schema export identity and strictness example. |
| matrix_diff.demo.json | Demo Matrix semantic diff report with diagnostic-only boundaries. |
| matrix_diff.demo.md | Demo Markdown rendering for Matrix diff interpretation. |
| amd_bound_sanity.demo.json | Demo AMD bound sanity report with existing-evidence refs and warnings. |
| amd_bound_sanity.demo.md | Demo Markdown rendering for AMD bound sanity interpretation. |

All paths are synthetic relative refs such as `out/v1_19_demo/...` and
`logs/demo-build.log`. Checksums are synthetic `sha256:` strings. The fixtures
intentionally omit raw stdout/stderr bodies, dataset payloads, real performance
numbers, and hardware-validation output.
