# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

import pytest
from pathlib import Path

from sol_execbench.core.scoring.hardware_calibration.builder import (
    CalibrationRequest,
    ClockLockRequiredError,
    run_calibration,
)
from sol_execbench.core.scoring.hardware_calibration.environment import GpuEnvironment
from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    HipProbe,
    ProbeExecution,
)
from sol_execbench.core.scoring.hardware_calibration.rocprof_compute import (
    ProfilerEnvironment,
)


def _probe() -> HipProbe:
    return HipProbe(
        compile_candidate=lambda _: "passed",
        execute_candidate=lambda key: ProbeExecution(
            (10.0,) * 7, "TFLOP/s" if key.kind == "compute" else "GB/s"
        ),
        check_correctness=lambda _, __: True,
        check_stability=lambda _, __: True,
    )


def _probe_with_states(states: dict[str, str]) -> HipProbe:
    """Return a probe that overrides evidence states for selected candidates."""
    return HipProbe(
        compile_candidate=lambda _: "passed",
        execute_candidate=lambda key: ProbeExecution(
            (10.0,) * 7, "TFLOP/s" if key.kind == "compute" else "GB/s"
        ),
        check_correctness=lambda key, _: (
            states.get(key.value, "measured") == "measured"
        ),
        check_stability=lambda _, __: True,
    )


def _locked_clock():
    class Clock:
        locked = False

        def lock(self) -> bool:
            self.locked = True
            return True

        def unlock(self) -> None:
            self.locked = False

        def observe_locked(self) -> bool:
            return self.locked

    return Clock()


def test_rdna4_validates_when_real_wmma_candidate_is_measured() -> None:
    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(0, "gfx1200"),
            hip_probe=_probe_with_states({"compute.matrix.bf16.bf16.wmma": "measured"}),
            clock_controller=_locked_clock(),
        )
    )

    assert artifact.validation_status == "validated"


def test_calibration_binds_explicit_source_revision() -> None:
    revision = "a" * 40
    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(0, "gfx1200"),
            hip_probe=_probe_with_states({"compute.matrix.bf16.bf16.wmma": "measured"}),
            clock_controller=_locked_clock(),
            source_revision=revision,
        )
    )

    assert artifact.metadata["source_revision"] == revision


def test_rdna4_remains_provisional_when_wmma_is_unknown() -> None:
    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(0, "gfx1200"),
            hip_probe=_probe_with_states({"compute.matrix.bf16.bf16.wmma": "unknown"}),
            clock_controller=_locked_clock(),
        )
    )

    assert artifact.validation_status == "provisional"


def test_missing_clock_adapter_is_collected_but_provisional() -> None:
    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(device=0, architecture="gfx942"),
            hip_probe=_probe(),
        )
    )

    assert artifact.collection_status == "collected"
    assert artifact.validation_status == "provisional"
    assert artifact.metadata["clock_lock_reason"] == "clock_lock_adapter_unavailable"


def test_request_without_injected_probe_uses_default_backend(monkeypatch) -> None:
    probe = _probe()
    monkeypatch.setattr(
        "sol_execbench.core.scoring.hardware_calibration.builder.default_hip_probe",
        lambda **_: probe,
    )

    artifact = run_calibration(
        CalibrationRequest(environment=GpuEnvironment(device=0, architecture="gfx942"))
    )

    assert all(candidate.state == "measured" for candidate in artifact.candidates)


def test_require_clock_lock_rejects_missing_adapter() -> None:
    with pytest.raises(ClockLockRequiredError):
        run_calibration(
            CalibrationRequest(
                environment=GpuEnvironment(device=0, architecture="gfx950"),
                hip_probe=_probe(),
                require_clock_lock=True,
            )
        )


def test_rdna4_lock_lifecycle_wraps_measurement() -> None:
    events: list[str] = []

    class Clock:
        def lock(self) -> bool:
            events.append("lock")
            return True

        def unlock(self) -> None:
            events.append("unlock")

        def observe_locked(self) -> bool:
            # pre=False, benchmark window=True, reset=False
            return len([event for event in events if event == "lock"]) > len(
                [event for event in events if event == "unlock"]
            )

    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(device=0, architecture="gfx1200"),
            hip_probe=_probe(),
            clock_controller=Clock(),
        )
    )

    assert events == ["lock", "unlock"]
    assert artifact.validation_status == "validated"


def test_rdna4_preserves_external_clock_lock() -> None:
    events: list[str] = []

    class Clock:
        def lock(self) -> bool:
            events.append("lock")
            return True

        def unlock(self) -> None:
            events.append("unlock")

        def observe_locked(self) -> bool:
            return True

    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(device=0, architecture="gfx1200"),
            hip_probe=_probe(),
            clock_controller=Clock(),
        )
    )

    assert events == []
    assert artifact.metadata["clock_observations"] == {
        "pre": True,
        "during": True,
        "post": True,
    }
    assert artifact.validation_status == "validated"


def test_unlock_failure_is_not_silently_accepted() -> None:
    class Clock:
        observations = iter((False, True, True))

        def lock(self) -> bool:
            return True

        def unlock(self) -> bool:
            return False

        def observe_locked(self) -> bool:
            return next(self.observations)

    with pytest.raises(ClockLockRequiredError, match="clock_unlock_failed"):
        run_calibration(
            CalibrationRequest(
                environment=GpuEnvironment(device=0, architecture="gfx1200"),
                hip_probe=_probe(),
                clock_controller=Clock(),
            )
        )


def test_reset_is_attempted_after_unsuccessful_lock() -> None:
    events: list[str] = []

    class Clock:
        def lock(self) -> bool:
            events.append("lock")
            return False

        def unlock(self) -> None:
            events.append("unlock")

        def observe_locked(self) -> bool:
            return False

    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(device=0, architecture="gfx1200"),
            hip_probe=_probe(),
            clock_controller=Clock(),
        )
    )

    assert events == ["lock", "unlock"]
    assert artifact.metadata["clock_observations"]["post"] is False


def test_strict_lock_failure_still_attempts_reset() -> None:
    events: list[str] = []

    class Clock:
        def lock(self) -> bool:
            events.append("lock")
            return False

        def unlock(self) -> None:
            events.append("unlock")

        def observe_locked(self) -> bool:
            return False

    with pytest.raises(ClockLockRequiredError, match="clock_lock_failed"):
        run_calibration(
            CalibrationRequest(
                environment=GpuEnvironment(device=0, architecture="gfx1200"),
                hip_probe=_probe(),
                clock_controller=Clock(),
                require_clock_lock=True,
            )
        )

    assert events == ["lock", "unlock"]


def _profiler_environment(tmp_path: Path) -> ProfilerEnvironment:
    return ProfilerEnvironment(
        state="measured",
        reason_code=None,
        tool_path=Path("/usr/bin/rocprof-compute"),
        tool_version="test",
        requirements_sha256="digest",
        venv_path=tmp_path / "venv",
        interpreter_path=tmp_path / "venv/bin/python",
    )


def test_profiler_metric_is_attributed_to_every_matching_candidate(
    tmp_path: Path,
) -> None:
    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(device=0, architecture="gfx942"),
            hip_probe=_probe(),
            profiler_environment=_profiler_environment(tmp_path),
            profiler_run=lambda *args, **kwargs: type(
                "Result", (), {"stdout": "Metric,Value\nPeak FP32,1\n"}
            )(),
        )
    )

    assert artifact.metadata["rocprof_compute_profile_status"] == "collected"
    assert artifact.metadata["rocprof_compute_recognised_candidate_keys"] == [
        "compute.vector.fp32.fp32.portable",
        "compute.vector.fp32.fp32.gfx94",
    ]


def test_empty_profiler_csv_is_not_recognised_or_collected(tmp_path: Path) -> None:
    artifact = run_calibration(
        CalibrationRequest(
            environment=GpuEnvironment(device=0, architecture="gfx942"),
            hip_probe=_probe(),
            profiler_environment=_profiler_environment(tmp_path),
            profiler_run=lambda *args, **kwargs: type("Result", (), {"stdout": ""})(),
        )
    )

    assert artifact.metadata["rocprof_compute_profile_status"] == "unknown"
    assert artifact.metadata["rocprof_compute_recognised_candidate_keys"] == []
