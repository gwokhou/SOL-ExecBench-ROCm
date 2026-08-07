from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sol_execbench.core.bench.performance_model.lifecycle import (
    DiagnosticRetentionClass,
    retention_duration_days,
    retention_reclaimable,
)

_NOW = datetime(2026, 8, 7, tzinfo=UTC)


def test_cache_is_immediately_age_reclaimable() -> None:
    assert retention_reclaimable(
        DiagnosticRetentionClass.CACHE,
        _NOW,
        _NOW,
    )


def test_debug_requires_the_bounded_window() -> None:
    recent = _NOW - timedelta(days=13)
    assert not retention_reclaimable(
        DiagnosticRetentionClass.DEBUG, recent, _NOW
    )
    aged = _NOW - timedelta(days=15)
    assert retention_reclaimable(DiagnosticRetentionClass.DEBUG, aged, _NOW)


def test_process_evidence_uses_the_grace_window() -> None:
    assert (
        retention_duration_days(DiagnosticRetentionClass.PROCESS_EVIDENCE)
        == 90.0
    )
    within = _NOW - timedelta(days=89)
    assert not retention_reclaimable(
        DiagnosticRetentionClass.PROCESS_EVIDENCE, within, _NOW
    )


def test_frozen_and_release_classes_are_never_age_reclaimable() -> None:
    for retention_class in (
        DiagnosticRetentionClass.FROZEN_SOURCE_EVIDENCE,
        DiagnosticRetentionClass.PUBLICATION_RELEASE,
    ):
        old = _NOW - timedelta(days=3650)
        assert not retention_reclaimable(retention_class, old, _NOW)
        assert retention_duration_days(retention_class) is None
