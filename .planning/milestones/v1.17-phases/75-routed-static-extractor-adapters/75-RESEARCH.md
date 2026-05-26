# Phase 75: Routed Static Extractor Adapters - Research

**Researched:** 2026-05-26  
**Domain:** Routed static extractor execution, bounded raw output preservation, and nonfatal sidecar semantics  
**Confidence:** HIGH for `llvm-objdump`/`readelf`; MEDIUM for future RGA/`roc-objdump`

<user_constraints>
## User Constraints (from CONTEXT.md)

- Execute only routed `llvm-objdump` and `readelf` adapters in Phase 75.
- Route extractor attempts through `src/sol_execbench/core/toolchain.py`; do
  not use ad hoc executable lookup.
- Keep `roc-objdump` and RGA as candidate/planned/unavailable route records
  only, not execution paths.
- Preserve bounded raw output artifacts for every extractor attempt.
- Record command provenance, timeout, return code, stdout/stderr tails, and raw
  output path in tool-run records.
- Treat missing, failed, timed-out, nonzero, and parser-failed tools as
  nonfatal diagnostic outcomes.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SKE-EXTRACT-01 | Route static extraction through v1.16 toolchain routing instead of direct ad hoc executable lookup. | `build_toolchain_routing_report()` supports injected `which`, bounded probe runner, and `ToolchainEvidenceLevel.STATIC`. [VERIFIED: `src/sol_execbench/core/toolchain.py`] |
| SKE-EXTRACT-02 | Run bounded `llvm-objdump` extraction when routing selects an available compatible tool. | `llvm-objdump` registry entry already exists for `ROCM_BINARY`, `ELF_OBJECT`, and `STATIC_FUTURE`; Phase 75 must promote lifecycle from `PLANNED` to an executable lifecycle. [VERIFIED: `src/sol_execbench/core/toolchain.py`] |
| SKE-EXTRACT-03 | Run bounded `readelf` metadata extraction as fallback or complementary route when available. | `readelf` registry entry already exists for `ROCM_BINARY`, `ELF_OBJECT`, and `STATIC_FUTURE`; Phase 75 can add a command builder for `readelf --headers --wide`. [VERIFIED: `src/sol_execbench/core/toolchain.py`] |
| SKE-EXTRACT-04 | Record route decisions, selected/unavailable tools, command provenance, timeout, return code, stdout/stderr tails, and raw output artifact paths. | `StaticKernelEvidenceToolRun` already records command, status, reason, return code, tails, and timeout; add raw output path. Routing decisions can be represented as source references/warnings and per-tool unavailable runs. [VERIFIED: `src/sol_execbench/core/bench/static_kernel_evidence.py`] |
| SKE-EXTRACT-05 | Preserve raw extractor output and derive conservative facts only. | Preserve bounded raw output text files, then set only `metadata_present`, `disassembly_present`, detected gfx strings, and rough symbol count when output supports it. [VERIFIED: planned tests] |
| SKE-EXTRACT-06 | Return nonfatal failed, partial, or unavailable sidecars for missing, unsupported, timed-out, nonzero, or parser-failed tools. | Existing static status/reason enums include `partial`, `unavailable`, `failed`, `toolchain_unavailable`, `extractor_failed`, `extractor_timeout`, and `parser_failed`. [VERIFIED: `src/sol_execbench/core/bench/static_kernel_evidence.py`] |

</phase_requirements>

## Summary

Phase 75 should add an extractor helper in `static_kernel_evidence.py` that
takes Phase 74 artifact entries, routes `llvm-objdump` and `readelf` through the
toolchain registry, runs only available compatible tools with bounded timeouts,
persists bounded raw output artifacts, and returns a diagnostic sidecar. It
should also update `toolchain.py` so `llvm-objdump` and `readelf` are active
routes rather than planned placeholders.

The minimum useful commands are:

- `llvm-objdump --disassemble <artifact>`
- `readelf --headers --wide <artifact>`

`readelf` is metadata-only, not ISA authority. `llvm-objdump` output can set
`disassembly_present=True` when nonempty. Both outputs may contribute
conservative architecture and symbol hints, but rich parsing belongs to later
work.

## Architecture Notes

### Routing

Use `build_toolchain_routing_report()` with a registry filtered to one target
tool per attempt. This keeps every executable decision inside the v1.16 routing
layer while allowing Phase 75 to execute both `llvm-objdump` and `readelf`
instead of only whichever tool appears first in the global registry.

### Runner Injection

Use two injectable dependencies:

- routing `which`/probe runner for tool availability tests
- extractor runner for command execution tests

This mirrors existing toolchain tests and keeps Phase 75 CPU-safe.

### Artifact Compatibility

Run extractors only for inspectable artifacts:

- `shared_library`, `hsaco`, and `code_object` -> `ToolchainArtifactType.ROCM_BINARY`
- `object_file` -> `ToolchainArtifactType.ELF_OBJECT`
- `compiler_output` -> unsupported for `llvm-objdump`/`readelf` execution

Unsupported artifacts should get nonfatal unsupported tool-run/warning entries,
not benchmark failures.

### Raw Output

Store bounded raw output under the evidence directory, for example:

```text
extractors/<artifact-id>/<tool-id>.txt
```

The stored text should include bounded stdout/stderr sections so callers can
inspect exactly what was available without embedding unlimited output in the
sidecar.

## Tests To Add

- `llvm-objdump` and `readelf` become active static routes and are selectable
  through injected routing probes.
- Available routed extractors run bounded commands and write raw output files.
- Tool-run records include command, timeout, return code, stdout/stderr tails,
  and raw output path.
- Missing tools return unavailable sidecars/tool-runs without failing.
- Nonzero extractor returns failed or partial sidecar depending on whether the
  companion tool succeeded.
- Timeout returns failed tool-run with `extractor_timeout`.
- Compiler-output artifacts are not executed by object extractors.

## Open Questions

None. User selected the conservative two-tool scope, bounded raw-output
preservation, and nonfatal partial/failure semantics.
