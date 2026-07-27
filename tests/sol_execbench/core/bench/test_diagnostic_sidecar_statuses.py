"""Regression coverage for the shared diagnostic-sidecar vocabulary."""

from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticFreshnessStatus,
    DiagnosticGovernanceStatus,
    DiagnosticSidecarStatus,
)


def test_shared_status_vocabularies_are_closed() -> None:
    assert {status.value for status in DiagnosticSidecarStatus} == {
        "available",
        "partial",
        "unavailable",
    }
    assert {status.value for status in DiagnosticFreshnessStatus} == {
        "current",
        "stale",
        "unknown",
    }
    assert {status.value for status in DiagnosticGovernanceStatus} == {
        "usable_diagnostic",
        "stale_diagnostic",
        "invalid_diagnostic",
        "unavailable",
    }
