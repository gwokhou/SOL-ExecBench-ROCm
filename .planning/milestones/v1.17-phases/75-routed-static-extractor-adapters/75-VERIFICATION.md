# Phase 75 Verification: Routed Static Extractor Adapters

**Verified:** 2026-05-26  
**Status:** PASS  
**Score:** 5/5

## Goal-Backward Result

Phase goal: operators can extract static ISA and metadata through routed,
bounded, nonfatal static-analysis tools.

Result: achieved. `llvm-objdump` and `readelf` are active routed static tools,
extractor execution is bounded and injectable, raw output artifacts are
preserved, and nonfatal diagnostic sidecars are returned for unavailable,
partial, failed, unsupported, and timed-out paths.

## Requirement Assessment

| Requirement | Verdict | Evidence |
|-------------|---------|----------|
| SKE-EXTRACT-01 | PASS | Each extractor route uses `build_toolchain_routing_report()` with `ToolchainEvidenceLevel.STATIC`. |
| SKE-EXTRACT-02 | PASS | `llvm-objdump --disassemble` executes only when routed available. |
| SKE-EXTRACT-03 | PASS | `readelf --headers --wide` executes only when routed available. |
| SKE-EXTRACT-04 | PASS | Tool-run payloads include route-derived status, command, timeout, return code, stdout/stderr tails, and raw output path. |
| SKE-EXTRACT-05 | PASS | Bounded raw output artifacts are saved; classification is limited to metadata/disassembly/architecture presence. |
| SKE-EXTRACT-06 | PASS | Tests cover missing tools, nonzero exits, timeouts, and unsupported compiler-output artifacts as nonfatal sidecar states. |

## Verification Commands

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py -q`
  - Result: 28 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_public_contract_guardrails.py tests/sol_execbench/test_trace_reporting_and_score_guardrails.py -q`
  - Result: 65 passed
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/bench/static_kernel_evidence.py src/sol_execbench/core/toolchain.py tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py`
  - Result: All checks passed

## Residual Risk

- CLI integration is not present until Phase 76.
- RGA and `roc-objdump` remain non-executed optional candidates until live
  command fixtures prove stable behavior.
- Raw-output parsing is intentionally conservative.

## Sign-Off

Phase 75 is complete and ready to transition to Phase 76.
