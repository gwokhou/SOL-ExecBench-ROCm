---
phase: 77
slug: documentation-guardrails-and-live-validation
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-26
---

# Phase 77 - Validation Strategy

## Commands

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_contract_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_static_kernel_evidence.py tests/sol_execbench/test_toolchain_routing.py tests/sol_execbench/test_cli_environment_snapshot.py tests/sol_execbench/test_research_release_docs.py tests/sol_execbench/test_public_contract_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check docs/CLAIMS.md docs/RESEARCHER-GUIDE.md docs/static_kernel_evidence.md docs/internal/v1_17_static_kernel_evidence_validation.md tests/sol_execbench/test_research_release_docs.py`

## Coverage

| Requirement | Automated Coverage |
|-------------|--------------------|
| SKE-DOCS-01 | Docs explain enablement, status vocabulary, and archived sidecars. |
| SKE-DOCS-02 | Claims doc states diagnostic-only boundaries. |
| SKE-DOCS-03 | Tests assert CPU-safe fixture/fake-tool coverage exists. |
| SKE-DOCS-04 | Internal validation artifact records live RDNA 4 attempt or explicit skip. |
| SKE-DOCS-05 | Docs list CDNA 3, CDNA 4, Triton, RGA-rich parsing, and paper-scale coverage as deferred unless evidenced. |
