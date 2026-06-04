# Phase 132: Local Dataset Migration Pipeline - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

Build deterministic, local-only migration tooling for user-downloaded
SOL-ExecBench and FlashInfer Trace inputs. The tooling may inspect and copy
local source files into a ROCm runner-compatible local layout, but it must not
add source dataset content to the repository or loosen Phase 131 redistribution
guardrails.

</domain>

<decisions>
## Implementation Decisions

### Local-Only Migration Boundary
- NVIDIA SOL-ExecBench source rows and generated migrated derivatives remain
  local-only. Migration outputs must record the `nvidia_sol_execbench` source
  boundary and must not be represented as redistributable project fixtures.
- FlashInfer Trace migration must use the distinct
  `flashinfer_flashinfer_trace` Apache-2.0 source boundary while preserving
  attribution metadata in generated manifests.
- Synthetic tests may create tiny local fixtures that mimic schema shape, but
  must not include external dataset payloads.

### Deterministic Output Contract
- Migration commands produce a runner-compatible layout with category/problem
  directories containing `definition.json`, `workload.jsonl`, optional
  `config.json`, optional solution files, and optional trace refs.
- Manifests are deterministic JSON sidecars with schema version, source dataset
  id, repo id, revision, checksums, generated artifact refs, license-boundary
  metadata, blocker records, and a stable manifest checksum.
- File ordering, JSON serialization, checksums, and blocker ordering must be
  deterministic for repeatable local runs.

### Blocker Semantics
- Missing required definition/workload/source rows are explicit blocking
  records.
- Missing optional blobs, safetensors refs, traces, and solution records are
  explicit blocker or warning states rather than silent omissions.
- Safetensors refs are never dereferenced outside the provided source root; a
  missing referenced blob is recorded as a blocker in the manifest.

### CLI Surface
- Add local migration commands under the existing `sol-execbench` CLI dispatch
  style. The command should support SOL-ExecBench and FlashInfer Trace sources
  separately, with source root, output root, source revision, and manifest path
  options.
- The command must be CPU-safe and network-free.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/REQUIREMENTS.md` — DATA-MIG-01 through DATA-MIG-04 requirements.
- `.planning/ROADMAP.md` — Phase 132 goal and success criteria.
- `.planning/phases/131-dataset-license-and-provenance-policy/131-CONTEXT.md`
  — Phase 131 license and redistribution decisions.
- `provenance.toml` — machine-readable source and redistribution policy.
- `src/sol_execbench/core/dataset/manifest.py` — existing deterministic
  manifest conventions.
- `src/sol_execbench/core/dataset/checksums.py` — stable checksum helpers.
- `src/sol_execbench/cli/main.py` — existing Click dispatch style.
- `scripts/check_dataset_redistribution.py` — restricted dataset guardrail.

</canonical_refs>

<code_context>
## Existing Code Insights

- Dataset package helpers already use Pydantic models, deterministic JSON
  serialization, and stable sha256 checksums.
- `scripts/run_dataset.py` consumes the local benchmark problem layout; Phase
  132 should generate that layout but not integrate runner behavior.
- The CLI currently dispatches `contract`, `doctor`, and `toolchain`
  subcommands through a custom `click.Command` wrapper.

</code_context>

<deferred>
## Deferred Ideas

- ROCm readiness classification and ready subset generation remain Phase 133.
- Low-precision compatibility abstractions remain Phase 134.
- Dataset runner consumption of migration manifests remains Phase 135.

</deferred>
