# Phase 81: Runtime Evidence And Compatibility Reports - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning
**Source:** $gsd-autonomous smart discuss, accepted by user

<domain>
## Phase Boundary

Users can collect scoped runtime evidence and review per-Target and aggregate
compatibility reports without changing benchmark semantics.

This phase covers requirements EVID-01 through EVID-06:

- Separate host ROCm/driver/device-node evidence from container ROCm user-space
  and toolchain evidence.
- Include Python runtime metadata: `torch.__version__`, `torch.version.hip`,
  `torch.version.cuda`, PyTorch device availability, and `triton-rocm` status.
- Include GPU metadata: device count, device name, detected `gfx*`
  architecture, and visible-device environment variables when available.
- Emit per-Target compatibility JSON and aggregate compatibility matrix JSON
  with status counts.
- Keep setup failures, dependency failures, and benchmark
  correctness/performance results as distinct evidence categories.
- Do not mutate canonical trace JSONL, correctness semantics, timing semantics,
  scoring schemas, benchmark defaults, or benchmark exit semantics.

</domain>

<decisions>
## Implementation Decisions

### Evidence scopes

- Record host, container, toolchain, Python dependency, dependency policy, and
  GPU evidence as separate scopes.
- Do not collapse requested Target values and observed evidence into one flat
  payload.

### Collection ownership

- Put structured evidence/report generation in Python core modules under
  `src/sol_execbench/core/`.
- Keep `scripts/run_docker.sh` as a thin wrapper that calls Python helpers and
  writes sidecars only through explicit opt-in paths.

### Host evidence

- Host probes may record Docker context/host, `/dev/kfd`, `/dev/dri`, and
  driver/ROCm summaries when available.
- Missing host probes should be diagnostic nullable fields, not failures by
  themselves. Existing preflight blockers remain blockers.

### Container and toolchain evidence

- Record requested container image/tag and observed container/toolchain values
  such as ROCm user-space version and `hipcc --version` when available.
- Container evidence must not claim native host validation.

### Python and dependency evidence

- Reuse Phase 80 dependency policy and dependency observation models where
  possible.
- Include torch distribution/runtime version, HIP version, CUDA compatibility
  namespace version, PyTorch device availability, torchvision, and
  `triton-rocm` status.

### GPU metadata

- Collect GPU count/name/gfx architecture/visible-device environment from
  PyTorch when import and device access are available.
- Support injected observations so tests are CPU-safe and do not require live
  ROCm hardware.

### Failure taxonomy

- Distinguish setup/runtime unavailable evidence, dependency failure evidence,
  and benchmark correctness/performance result evidence.
- Do not turn setup or dependency failure into benchmark correctness failure.

### Report artifacts

- Emit per-Target compatibility JSON sidecars and aggregate compatibility matrix
  JSON with `status_counts`.
- Keep aggregate schema compatible with
  `sol_execbench.rocm_compatibility_matrix.v1`.

### Benchmark semantics

- Compatibility evidence remains diagnostic sidecar data only.
- Do not change trace JSONL, scoring/timing/correctness schemas, benchmark
  defaults, or default exit semantics.

</decisions>

<code_context>
## Existing Code Insights

- `src/sol_execbench/core/compatibility.py` already defines strict
  compatibility matrix models, Matrix Entry status/reason vocabularies, evidence
  scopes, claim boundaries, and `RocmCompatibilityMatrixReport`.
- `src/sol_execbench/core/dependency_matrix.py` already collects PyTorch ROCm
  dependency evidence and classifies dependency preflight results.
- `src/sol_execbench/core/docker_matrix.py` already maps declared Docker Targets
  to Matrix Target payloads and Docker runtime preflight entries.
- `scripts/run_docker.sh` already performs declared target selection, dependency
  preflight, Docker runtime preflight, and dry-run behavior.
- `src/sol_execbench/cli/main.py` already writes optional diagnostic
  environment and profiling sidecars without mutating canonical trace output.

</code_context>

<specifics>
## Specific Ideas

- Add a runtime report helper module that can:
  - collect host/toolchain/GPU evidence,
  - convert dependency and Docker preflight entries into per-target sidecars,
  - build aggregate reports from one or more Matrix Entry payloads,
  - write JSON deterministically.
- Add a small CLI under `python -m sol_execbench.core.<module>` for sidecar
  generation and aggregation.
- Add explicit environment variables or wrapper flags for Docker sidecar output,
  while preserving default script behavior.
- Add CPU-safe tests for model output, aggregation counts, sidecar writing, and
  non-mutation of canonical trace payloads.

</specifics>

<deferred>
## Deferred Ideas

- Live RDNA4/CDNA3 hardware validation is out of scope for this phase.
- Documentation of target/requested-vs-observed interpretation belongs to the
  follow-up docs phase unless a small implementation note is necessary.

</deferred>
