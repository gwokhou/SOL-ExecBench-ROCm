# Phase 53: Dataset Contract And Acquisition Metadata - Context

**Gathered:** 2026-05-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase establishes the public SOL-ExecBench dataset acquisition and
local-layout contract for `L1`, `L2`, `Quant`, and `FlashInfer-Bench`. It
verifies or downloads dataset layout, emits acquisition/local-layout metadata,
and makes acquisition status machine-readable without claiming ROCm readiness,
execution success, paper validation, hosted leaderboard parity, or upstream
SOLAR equivalence.

</domain>

<decisions>
## Implementation Decisions

### Dataset Source And Layout Contract
- Model the canonical dataset root as
  `data/SOL-ExecBench/benchmark/{L1,L2,Quant,FlashInfer-Bench}` to match the
  current docs and dataset-runner examples.
- Default downloader output should be `data/SOL-ExecBench/benchmark`, with
  `--output-root` support and downloaded contents kept out of committed source
  files.
- Acquisition/local-layout manifests should record Hugging Face repository ID,
  subset/category, revision or local provenance, root path, discovered counts,
  and checksum metadata.
- Missing category directories should produce structured `missing_category`
  diagnostics and fail unless the user explicitly selected a partial category
  set.

### Downloader And Idempotency
- Keep downloader changes narrow: add category selection, output-root,
  manifest output, and idempotent checks without rewriting the download logic.
- Existing files and directories should be reused and layout-verified by
  default; do not delete or overwrite unknown files.
- Dependency, Hugging Face access, or network failures should produce clear
  errors while preserving the local-layout manifest path when applicable.
- Category selection should use repeatable `--category L1 --category Quant`
  flags, with all four public categories selected by default.

### Artifact Shape And Integration Boundary
- Add reusable dataset contract/manifest code under
  `src/sol_execbench/core/dataset/`; scripts should stay thin CLIs over that
  library code.
- Stabilize manifest output through typed internal models and deterministic
  JSON. Keep fields conservative and sidecar-only.
- Do not modify public benchmark schemas:
  `definition.json`, `workload.jsonl`, `solution.json`, and trace JSONL remain
  unchanged.
- Write manifests to an explicit artifact/output path such as
  `dataset_manifest.json`; do not write into dataset roots unless explicitly
  requested.

### Verification And Claim Boundary
- Test layout/category validation, manifest/checksum generation, idempotency,
  and downloader CLI behavior with fixtures and mocks; do not require real
  network access.
- Documentation should explicitly state that acquisition/layout completion is
  not readiness, execution, or paper-level validation.
- Tests may pass without a real local dataset by using temporary fixture
  directories. Real dataset checks are user-run commands.
- Phase 53 must not produce ready subsets or readiness classifications; those
  belong to Phase 54.

### the agent's Discretion
The agent may choose exact helper names, manifest field ordering, checksum
algorithm details, and CLI option grouping as long as the public behavior above
is preserved and follows existing repository style.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/download_solexecbench.py` already downloads `nvidia/SOL-ExecBench`
  via `datasets.load_dataset` and writes benchmark assets.
- `scripts/download_data.sh` orchestrates SOL-ExecBench and FlashInfer trace
  downloads.
- `scripts/run_dataset.py` already discovers category subdirectories and runs
  dataset batches against the primary `sol-execbench` CLI.
- `src/sol_execbench/core/data/` provides the public Pydantic benchmark
  contracts that future inventory phases will use.

### Established Patterns
- Keep reusable package logic under `src/sol_execbench/core/` and leave scripts
  as operational entry points.
- Preserve canonical benchmark input/output schemas and place derived metadata
  in sidecar artifacts.
- Use deterministic JSON/JSONL shapes for machine-readable outputs and clear
  claim-boundary wording in docs and tests.
- Use Python 3.12 typing, small module-private helpers, and focused pytest
  fixtures.

### Integration Points
- `scripts/download_solexecbench.py` is the narrowest existing entry point for
  downloader idempotency and category/output-root improvements.
- A new `src/sol_execbench/core/dataset/` package can expose layout and
  manifest helpers for Phase 54 inventory/readiness work.
- Documentation updates should live near dataset setup guidance in
  `docs/GETTING-STARTED.md`, `docs/analysis.md`, or a focused dataset contract
  page if the implementation warrants it.

</code_context>

<specifics>
## Specific Ideas

Keep the milestone aligned with the paper's public dataset surface while being
strict about claim boundaries: this phase proves acquisition/layout metadata,
not ROCm execution parity.

</specifics>

<deferred>
## Deferred Ideas

- Ready-subset generation and ROCm readiness classification are deferred to
  Phase 54.
- Bounded execution closure is deferred to Phase 55.
- Parity gap reporting is deferred to Phase 56.
- Full claim-guardrail release wording and milestone closure are deferred to
  Phase 57.

</deferred>
