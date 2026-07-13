---
status: complete
completed: 2026-05-31
---

# Fix v1.20 Audit Gaps Summary

Fixed all blockers from `.planning/v1.20-MILESTONE-AUDIT.md`:

- Wired AMD SOL and SOLAR derivation evidence into claim-upgrade and trust-summary core builders, scripts, source refs, checksums, and tests.
- Added a CPU-safe four-script E2E regression for consistency -> stability -> claim-upgrade -> trust-summary.
- Added a consistent v1.20 fixture and required it in docs tests.
- Added phase `VERIFICATION.md` and `VALIDATION.md` artifacts for phases 89-93.
- Updated the milestone audit report to `passed`.

Verification:

- `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/sol_execbench/test_consistency_report.py tests/sol_execbench/test_consistency_script.py tests/sol_execbench/test_evaluation_stability.py tests/sol_execbench/test_evaluation_stability_script.py tests/sol_execbench/test_claim_upgrade.py tests/sol_execbench/test_claim_upgrade_script.py tests/sol_execbench/test_trust_summary.py tests/sol_execbench/test_trust_summary_script.py tests/sol_execbench/test_v1_20_evidence_quality_docs.py tests/sol_execbench/test_v1_20_evidence_quality_e2e.py tests/sol_execbench/test_public_contract_guardrails.py -q`
- `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/sol_execbench/core/claim_upgrade.py src/sol_execbench/core/trust_summary.py scripts/report_claim_upgrade.py scripts/report_trust_summary.py tests/sol_execbench/test_claim_upgrade.py tests/sol_execbench/test_trust_summary.py tests/sol_execbench/test_v1_20_evidence_quality_docs.py tests/sol_execbench/test_v1_20_evidence_quality_e2e.py docs/internal/v1_20_evidence_quality_guide.md`

