# Phase 76: CLI Sidecar Integration And Reports - Pattern Map

**Mapped:** 2026-05-26

## Patterns

- Use `_profile_output_directory()` / `_profile_sidecar_path()` /
  `_write_profile_sidecar()` as path/write analogs.
- Use `click.Choice([STATIC_EVIDENCE_NONE, STATIC_EVIDENCE_AUTO])` for the
  public flag, matching the profiler option style.
- Keep collection helpers pure enough for direct unit tests.
- Update public contract guardrails to allow `--static-evidence` only now that
  Phase 76 owns the public surface.

## Anti-Patterns

- No static evidence fields in trace JSONL.
- No static evidence effect on evaluation exit code.
- No global cache search.
- No Markdown report generation in this phase.
