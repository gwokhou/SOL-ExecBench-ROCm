# Requirements: SOL ExecBench ROCm Port v1.17

**Defined:** 2026-05-25
**Milestone:** v1.17 Static Kernel Evidence
**Core Value:** Evaluate LLM-generated GPU kernels correctly and reproducibly
on AMD ROCm hardware while preserving the benchmark semantics and rigor of SOL
ExecBench.

## v1.17 Requirements

Requirements for this milestone. Each requirement must map to exactly one
roadmap phase.

### Contract

- [x] **SKE-CONTRACT-01**: Maintainer can serialize and parse a strict
  `sol_execbench.static_kernel_evidence.v1` sidecar schema with stable status,
  reason-code, artifact, tool-run, and classification fields.
- [x] **SKE-CONTRACT-02**: Maintainer can represent static evidence authority
  boundaries in the sidecar, with diagnostic-only semantics and explicit false
  authority for correctness, performance, timing, score, paper parity, and
  leaderboard claims.
- [x] **SKE-CONTRACT-03**: Maintainer can record aggregate and per-artifact
  static evidence states for collected, partial, unavailable, unsupported,
  failed, and skipped outcomes.
- [x] **SKE-CONTRACT-04**: Consumer can discover static evidence support
  through evaluator contract optional capability metadata without changing the
  required evaluator contract version.
- [x] **SKE-CONTRACT-05**: Maintainer can verify that static evidence models
  and sidecar helpers do not mutate canonical trace JSONL, correctness, timing,
  scoring, or default benchmark behavior.

### Artifacts

- [x] **SKE-ARTIFACT-01**: Operator can collect static evidence only from the
  current HIP/C++ staging or build tree, starting with `benchmark_kernel.so`
  and opportunistically including code objects, HSACO files, object files, and
  compiler outputs exposed by that build.
- [x] **SKE-ARTIFACT-02**: Operator can persist discovered static artifacts
  into an output-derived evidence directory before temporary staging cleanup.
- [x] **SKE-ARTIFACT-03**: Operator can inspect an artifact manifest that
  records artifact kind, source path, persisted path, SHA256, size, producer,
  target architecture when known, and inspectability for every registered
  artifact.
- [x] **SKE-ARTIFACT-04**: Maintainer can verify that static artifact discovery
  avoids global cache, ROCm install tree, and unrelated temporary-directory
  scans that could mix artifacts across benchmark runs.
- [x] **SKE-ARTIFACT-05**: Operator receives explicit unsupported or
  unavailable states for solution paths without a stable v1.17 static artifact
  boundary, including Python/PyTorch eager, opaque external library, and
  unstable Triton-cache cases.

### Extractors

- [x] **SKE-EXTRACT-01**: Operator can route static extraction through the
  v1.16 toolchain routing layer instead of direct ad hoc executable lookup.
- [x] **SKE-EXTRACT-02**: Operator can run bounded `llvm-objdump` static
  extraction when routing selects an available compatible tool.
- [x] **SKE-EXTRACT-03**: Operator can run bounded `readelf` metadata
  extraction as a fallback or complementary static route when available.
- [x] **SKE-EXTRACT-04**: Operator can see route decisions, selected tools,
  unavailable tools, command provenance, timeout, return code, stdout/stderr
  tails, and raw output artifact paths for every attempted extractor.
- [x] **SKE-EXTRACT-05**: Maintainer can preserve raw extractor output and
  derive only conservative normalized facts such as detected architecture,
  section or code-object presence, symbol inventory, ISA-output presence, and
  metadata-output presence.
- [x] **SKE-EXTRACT-06**: Operator receives nonfatal failed, partial, or
  unavailable sidecars when tools are missing, unsupported, time out, return
  nonzero, or produce unparseable output.

### CLI And Reports

- [x] **SKE-CLI-01**: Operator can opt into static evidence collection from the
  benchmark CLI with `--static-evidence auto`, while `--static-evidence none`
  remains the default.
- [x] **SKE-CLI-02**: Operator can find the static evidence JSON sidecar and
  evidence artifact directory beside the configured trace output path.
- [x] **SKE-CLI-03**: Operator can run benchmark evaluation normally when
  static evidence collection is skipped, unavailable, partial, or failed; static
  evidence must not change benchmark exit-code semantics except for invalid CLI
  usage.
- [x] **SKE-CLI-04**: Researcher can read a human-facing static evidence report
  or summary that explains aggregate status, artifact manifest, routing
  decisions, extracted metadata/ISA presence, unsupported states, and claim
  boundaries.

### Documentation And Validation

- [x] **SKE-DOCS-01**: Researcher can read documentation that explains how to
  enable static evidence, interpret collected, partial, unavailable,
  unsupported, failed, and skipped states, and archive static sidecars.
- [x] **SKE-DOCS-02**: Researcher can read claim-boundary documentation stating
  that static evidence is diagnostic static-analysis evidence, not correctness,
  performance, timing, score, paper-parity, or leaderboard authority.
- [x] **SKE-DOCS-03**: Maintainer can run CPU-safe tests with fixture artifacts,
  fake routed tools, and fake extractor outputs for schema, artifact discovery,
  extractor parsing, CLI sidecar writing, and claim guardrails.
- [x] **SKE-DOCS-04**: Maintainer can record a bounded live ROCm validation
  artifact for at least one RDNA 4 HIP/C++ build when the execution environment
  exposes the required ROCm build tools and device/runtime access.
- [x] **SKE-DOCS-05**: Maintainer can document CDNA 3, CDNA 4, Triton, RGA-rich
  resource parsing, and paper-scale static coverage as unsupported, partial, or
  deferred unless this milestone produces direct evidence for those scopes.

## Future Requirements

Deferred to future milestones unless explicitly pulled forward by a roadmap
revision.

### Rich Static Analysis

- **SKE-FUTURE-01**: Researcher can inspect RGA-derived VGPR, SGPR, LDS,
  scratch, occupancy-like, or resource-summary metrics through stable parser
  fixtures and live tool validation.
- **SKE-FUTURE-02**: Researcher can inspect richer ISA instruction-family
  classifications and static diff reports between two sidecars for the same
  solution.
- **SKE-FUTURE-03**: Researcher can correlate static sidecar kernel symbols
  with optional `rocprofv3` profile kernel activity as diagnostic hints.

### Broader Artifact Coverage

- **SKE-FUTURE-04**: Operator can collect stable static evidence for Triton
  ROCm-generated kernels once cache-to-solution provenance is proven reliable.
- **SKE-FUTURE-05**: Operator can aggregate static evidence status across a
  dataset run without requiring full paper-scale coverage.
- **SKE-FUTURE-06**: Operator can run a standalone static artifact analysis
  command against an existing `.hsaco`, code object, shared object, or ELF
  artifact.

## Out of Scope

Explicitly excluded from v1.17.

| Feature | Reason |
|---------|--------|
| Static evidence as correctness authority | Static inspection cannot prove numerical correctness; canonical benchmark evaluation remains authoritative. |
| Static evidence as timing, performance, or score authority | ISA and metadata are diagnostic context, not measured latency or AMD SOL/SOLAR score inputs. |
| Mandatory static evidence for every benchmark run | Tool and artifact availability vary by ROCm install, solution type, and build path. |
| Full paper-scale 235-problem static coverage | The milestone targets a per-run evidence surface, not complete paper-scale coverage. |
| RGA-rich resource parsing as a required deliverable | RGA availability and output shape require live fixture validation; v1.17 starts with route-aware optional support. |
| Triton cache capture as a required deliverable | Stable provenance from generated cache artifacts to a benchmark solution is not yet proven. |
| Recompiling solutions with different optimization or artifact flags by default | Changing compile behavior can invalidate benchmark comparability. |
| Canonical trace JSONL schema changes | Static evidence is sidecar-only and must not alter the primary trace contract. |
| Hosted leaderboard or submission policy changes | v1.17 is a diagnostic evidence milestone, not a public service or ranking milestone. |
| CDNA 3 or CDNA 4 hardware validation claims | Static schema/routing support is separate from real-hardware validation evidence. |
| CDNA 3 / MI300X real-hardware validation | Deferred to a dedicated hardware-validation milestone; v1.17 does not claim full adapted-suite validation on those targets. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SKE-CONTRACT-01 | Phase 73 | Complete |
| SKE-CONTRACT-02 | Phase 73 | Complete |
| SKE-CONTRACT-03 | Phase 73 | Complete |
| SKE-CONTRACT-04 | Phase 73 | Complete |
| SKE-CONTRACT-05 | Phase 73 | Complete |
| SKE-ARTIFACT-01 | Phase 74 | Complete |
| SKE-ARTIFACT-02 | Phase 74 | Complete |
| SKE-ARTIFACT-03 | Phase 74 | Complete |
| SKE-ARTIFACT-04 | Phase 74 | Complete |
| SKE-ARTIFACT-05 | Phase 74 | Complete |
| SKE-EXTRACT-01 | Phase 75 | Complete |
| SKE-EXTRACT-02 | Phase 75 | Complete |
| SKE-EXTRACT-03 | Phase 75 | Complete |
| SKE-EXTRACT-04 | Phase 75 | Complete |
| SKE-EXTRACT-05 | Phase 75 | Complete |
| SKE-EXTRACT-06 | Phase 75 | Complete |
| SKE-CLI-01 | Phase 76 | Complete |
| SKE-CLI-02 | Phase 76 | Complete |
| SKE-CLI-03 | Phase 76 | Complete |
| SKE-CLI-04 | Phase 76 | Complete |
| SKE-DOCS-01 | Phase 77 | Complete |
| SKE-DOCS-02 | Phase 77 | Complete |
| SKE-DOCS-03 | Phase 77 | Complete |
| SKE-DOCS-04 | Phase 77 | Complete |
| SKE-DOCS-05 | Phase 77 | Complete |

**Coverage:**
- v1.17 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0

---
*Requirements defined: 2026-05-25*
*Last updated: 2026-05-26 after v1.17 completion*
