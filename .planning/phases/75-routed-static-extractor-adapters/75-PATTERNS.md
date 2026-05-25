# Phase 75: Routed Static Extractor Adapters - Pattern Map

**Mapped:** 2026-05-26  
**Files analyzed:** 5  
**Analogs found:** 5 / 5

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/sol_execbench/core/bench/static_kernel_evidence.py` | utility, model | routed command execution, sidecar serialization | `src/sol_execbench/core/bench/rocm_profiler.py`; existing static evidence helpers | high |
| `src/sol_execbench/core/toolchain.py` | registry, routing | request-response route decisions | existing toolchain routing code | exact |
| `tests/sol_execbench/test_static_kernel_evidence.py` | test | injected runners, temp files | `tests/sol_execbench/test_rocm_profiler.py`; existing static evidence tests | high |
| `tests/sol_execbench/test_toolchain_routing.py` | test | injected which/probe runner | existing toolchain routing tests | exact |

## Pattern Assignments

### Static Extractor Helper

Follow `rocm_profiler.py` for bounded command execution and output tailing, but
return strict Pydantic sidecar models from `static_kernel_evidence.py`.

Use injected runners in tests. The production runner may use `subprocess.run`
with `check=False`, `capture_output=True`, `text=True`, and `timeout=...`.

### Toolchain Routing

Reuse `build_toolchain_routing_report()` for every extractor attempt. Do not
call `shutil.which()` directly from the extractor helper except through the
toolchain routing API dependency injection.

Phase 75 should update tests that previously described static tools as planned
v1.16 placeholders. `llvm-objdump` and `readelf` become executable active
routes; RGA and `roc-objdump` remain optional/candidate/planned records.

### Raw Output Artifacts

Use deterministic evidence-relative paths:

```text
extractors/<artifact-id>/<tool-id>.txt
```

Keep file contents bounded and record the path on the tool-run. Do not embed
large raw outputs in the sidecar.

## Anti-Patterns To Avoid

- No direct ad hoc executable lookup in extractor code.
- No mandatory RGA or `roc-objdump`.
- No unbounded stdout/stderr persistence.
- No static evidence effect on trace, scoring, timing, or default CLI behavior.
