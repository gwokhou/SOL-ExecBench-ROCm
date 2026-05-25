# Phase 59: Benchmark Run Evidence Integration - Context

**Gathered:** 2026-05-25
**Status:** Ready for planning
**Mode:** Autonomous

<domain>
## Phase Boundary

Phase 59 attaches the Phase 58 environment snapshot contract to benchmark run
artifacts without changing canonical trace JSONL, correctness, timing, scoring,
or existing CLI defaults.

The integration must be explicitly enabled and sidecar-based. Phase 60 owns the
public diagnostic CLI and preflight command surface.
</domain>

<decisions>
## Implementation Decisions

### Compatibility
- Do not add fields to `Trace`, `Evaluation`, or canonical trace JSONL.
- Do not add new root benchmark CLI options in Phase 59.
- Do not collect environment snapshots unless explicitly enabled by a
  non-default environment variable or an explicit sidecar path.
- Environment snapshot failures must not affect benchmark exit status.

### Integration Shape
- Add CLI-internal helper logic that writes a JSON sidecar when enabled.
- Prefer `SOLEXECBENCH_ENV_SNAPSHOT_PATH` for explicit output location.
- Allow `SOLEXECBENCH_ENV_SNAPSHOT=1` to write beside `--output` using a stable
  suffix.
- Collect outside the measured eval-driver timing window.
</decisions>

<canonical_refs>
## Canonical References

- `.planning/milestones/v1.13-REQUIREMENTS.md`
- `.planning/phases/58-environment-snapshot-contract/58-VERIFICATION.md`
- `src/sol_execbench/core/environment.py`
- `src/sol_execbench/cli/main.py`
- `tests/sol_execbench/test_public_contract_guardrails.py`
- `tests/sol_execbench/test_environment_snapshot.py`
</canonical_refs>

<deferred>
## Deferred Ideas

- `sol-execbench doctor` / `env-snapshot` public command is Phase 60.
- Profiling artifacts are v1.14.
</deferred>

