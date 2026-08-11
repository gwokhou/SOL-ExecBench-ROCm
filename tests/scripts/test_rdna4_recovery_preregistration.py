from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sol_execbench.core.bench.performance_model.source_transition import (
    SourceTransitionDisposition,
    SourceTransitionStage,
)
from sol_execbench.core.bench.performance_model.vram_policy import (
    select_vram_working_set_policy,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file


def _prior(policy_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        base_source_revision="1" * 40,
        target_source_revision="2" * 40,
        stage_decisions=tuple(
            SimpleNamespace(
                stage=stage,
                disposition=SourceTransitionDisposition.UNCHANGED,
            )
            for stage in SourceTransitionStage
        ),
        reusable_artifacts=(
            SimpleNamespace(
                relative_path="vram-policy.json",
                sha256=sha256_file(policy_path),
            ),
        ),
    )


def _review(*, calibration_changed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        base_source_revision="2" * 40,
        target_source_revision="3" * 40,
        affects=lambda stage: (
            calibration_changed and stage is SourceTransitionStage.CALIBRATION
        ),
    )


def test_recovery_policy_reuse_requires_continuous_unchanged_chain(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovery = load_script(
        "scripts/internal/rdna4/preregister_rdna4_recovery.py"
    )
    policy = select_vram_working_set_policy(
        gpu_architecture="gfx1200",
        gpu_id="gpu-1",
        total_memory_bytes=16 * (1 << 30),
        source_revision="1" * 40,
        created_at="2026-08-12T00:00:00+00:00",
    )
    policy_path = tmp_path / "vram-policy.json"
    atomic_write_json_value(policy_path, policy.model_dump(mode="json"))
    monkeypatch.setattr(
        recovery,
        "load_and_verify_source_review",
        lambda *_args, **_kwargs: _review(),
    )

    observed = recovery._require_reusable_policy_chain(
        policy_path=policy_path,
        source_revision="3" * 40,
        prior=_prior(policy_path),
        review_path=tmp_path / "review.json",
    )

    assert observed == policy


def test_recovery_policy_reuse_rejects_calibration_change(
    load_script,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recovery = load_script(
        "scripts/internal/rdna4/preregister_rdna4_recovery.py"
    )
    policy = select_vram_working_set_policy(
        gpu_architecture="gfx1200",
        gpu_id="gpu-1",
        total_memory_bytes=16 * (1 << 30),
        source_revision="1" * 40,
        created_at="2026-08-12T00:00:00+00:00",
    )
    policy_path = tmp_path / "vram-policy.json"
    atomic_write_json_value(policy_path, policy.model_dump(mode="json"))
    monkeypatch.setattr(
        recovery,
        "load_and_verify_source_review",
        lambda *_args, **_kwargs: _review(calibration_changed=True),
    )

    with pytest.raises(
        ValueError, match="policy/calibration semantics changed"
    ):
        recovery._require_reusable_policy_chain(
            policy_path=policy_path,
            source_revision="3" * 40,
            prior=_prior(policy_path),
            review_path=tmp_path / "review.json",
        )
