"""Bind LayerNorm holdout workers to h0 commitment custody and rotations."""

from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import PRIVATE_FILE
from scripts.internal.solar.e2_layernorm_dev_inputs import (
    FEATURE_SHIFTS,
    rotate_inputs,
)
from scripts.internal.solar.e2_successor_capture import (
    SpoolingObservationService,
    instrument_spooling_drivers,
)
from scripts.internal.solar.e2_successor_exposure import (
    protect_observation_process,
)
from sol_execbench.core.bench.judge.contracts import JudgeInstance, JudgeTask
from sol_execbench.core.bench.reference_service import serve_reference_worker
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.process.termination import cleanup_on_termination

_SHIFT_FIELD = "feature_shift"
_SPOOL_IMPORT = (
    "from scripts.internal.solar.e2_successor_capture import "
    "serve_spooling_worker as serve_reference_worker"
)
_HOLDOUT_IMPORT = (
    "from scripts.internal.solar.e2_layernorm_holdout_inputs import "
    "serve_layernorm_holdout_worker as serve_reference_worker"
)


class LayerNormHoldoutService(SpoolingObservationService):
    """Enforce h0 commitment checks before applying registered rotations."""

    def __init__(self, staging_dir: Path, **kwargs: Any) -> None:
        """Accept only LayerNorm h0 coordinates and registered shifts."""
        control = load_json_value(staging_dir / PRIVATE_FILE)
        if (
            control["task"] != JudgeTask.LAYERNORM
            or control["instance"] != JudgeInstance.HOLDOUT
        ):
            raise ValueError("the LayerNorm holdout activity accepts h0 only")
        shift = control[_SHIFT_FIELD]
        if type(shift) is not int or shift not in FEATURE_SHIFTS:
            raise ValueError("unregistered LayerNorm holdout feature shift")
        self.feature_shift = shift
        super().__init__(staging_dir, **kwargs)

    def prepare_inputs(
        self,
        workload: Workload,
        row_index: int,
        round_index: int,
        *,
        variation_index: int | None = None,
    ) -> list[Any]:
        """Rotate validated h0 inputs identically for both sides."""
        actual = super().prepare_inputs(
            workload, row_index, round_index, variation_index=variation_index
        )
        if not self.feature_shift:
            return actual
        if self.latest_inputs is None:
            raise RuntimeError("the validated holdout input is missing")
        self.latest_inputs = rotate_inputs(
            self.latest_inputs, self.feature_shift
        )
        return [
            *(value.roll(self.feature_shift, dims=-1) for value in actual[:3]),
            actual[3],
        ]


def serve_layernorm_holdout_worker(
    staging_dir: str | Path, **kwargs: Any
) -> None:
    """Protect observation custody and run the h0-committed worker."""
    protect_observation_process()
    with cleanup_on_termination():
        serve_reference_worker(
            staging_dir, service_type=LayerNormHoldoutService, **kwargs
        )


def instrument_holdout(
    output: Path,
    *,
    shift: int,
    oracle_cpu_allowance_seconds: float,
    holdout_commitment_sha256: str,
) -> None:
    """Bind the h0 commitment and feature shift in worker-private state."""
    instrument_spooling_drivers(
        output,
        oracle_cpu_allowance_seconds=oracle_cpu_allowance_seconds,
        holdout_commitment_sha256=holdout_commitment_sha256,
    )
    worker = output / "reference_worker.py"
    source = worker.read_text()
    if source.count(_SPOOL_IMPORT) != 1:
        raise ValueError("LayerNorm holdout worker staging seam changed")
    worker.write_text(source.replace(_SPOOL_IMPORT, _HOLDOUT_IMPORT))
    private = output / PRIVATE_FILE
    control = load_json_value(private)
    atomic_write_json_value(private, {**control, _SHIFT_FIELD: shift})
    private.chmod(0o600)
