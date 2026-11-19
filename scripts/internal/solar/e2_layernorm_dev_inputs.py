"""Apply the same registered feature rotation to LayerNorm data and affine."""

from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import PRIVATE_FILE
from scripts.internal.solar.e2_successor_capture import (
    SpoolingObservationService,
    instrument_spooling_drivers,
)
from scripts.internal.solar.e2_successor_exposure import (
    protect_observation_process,
)
from sol_execbench.core.bench.judge.contracts import JudgeTask
from sol_execbench.core.bench.judge.inputs import JudgeInputs
from sol_execbench.core.bench.reference_service import serve_reference_worker
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.process.termination import cleanup_on_termination

FEATURE_SHIFTS = (0, 1, 17)
_SHIFT_FIELD = "feature_shift"
_SPOOL_IMPORT = (
    "from scripts.internal.solar.e2_successor_capture import "
    "serve_spooling_worker as serve_reference_worker"
)
_DEV_IMPORT = (
    "from scripts.internal.solar.e2_layernorm_dev_inputs import "
    "serve_layernorm_development_worker as serve_reference_worker"
)


def rotate_inputs(inputs: JudgeInputs, shift: int) -> JudgeInputs:
    """Rotate the actual FP32 features and carry their original permutation."""
    if shift not in FEATURE_SHIFTS:
        raise ValueError("unregistered LayerNorm development feature shift")
    if shift == 0:
        return inputs
    x, weight, bias, eps = inputs.values
    permutation = inputs.construction["feature_permutation"]
    return replace(
        inputs,
        values=(
            x.roll(shift, dims=-1),
            weight.roll(shift, dims=-1),
            bias.roll(shift, dims=-1),
            eps,
        ),
        construction={
            **inputs.construction,
            "feature_permutation": permutation[-shift:] + permutation[:-shift],
        },
    )


class LayerNormDevelopmentService(SpoolingObservationService):
    """Use the registered rotation only after validating the original input."""

    def __init__(self, staging_dir: Path, **kwargs: Any) -> None:
        """Reject holdout coordinates and bind the worker-private rotation."""
        control = load_json_value(staging_dir / PRIVATE_FILE)
        if control["task"] != JudgeTask.LAYERNORM or control[
            "instance"
        ] not in {"d0", "d1"}:
            raise ValueError(
                "the LayerNorm GPU activity accepts development only"
            )
        shift = control[_SHIFT_FIELD]
        if type(shift) is not int or shift not in FEATURE_SHIFTS:
            raise ValueError("unregistered LayerNorm development feature shift")
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
        """Apply identical rotations before either side sees the input."""
        actual = super().prepare_inputs(
            workload, row_index, round_index, variation_index=variation_index
        )
        if not self.feature_shift:
            return actual
        if self.latest_inputs is None:
            raise RuntimeError("the validated development input is missing")
        self.latest_inputs = rotate_inputs(
            self.latest_inputs, self.feature_shift
        )
        return [
            *(value.roll(self.feature_shift, dims=-1) for value in actual[:3]),
            actual[3],
        ]


def serve_layernorm_development_worker(
    staging_dir: str | Path, **kwargs: Any
) -> None:
    """Observe candidate exit and retain the permuted evidence in private."""
    protect_observation_process()
    with cleanup_on_termination():
        serve_reference_worker(
            staging_dir, service_type=LayerNormDevelopmentService, **kwargs
        )


def instrument_development(
    output: Path, *, shift: int, oracle_cpu_allowance_seconds: float
) -> None:
    """Bind the permutation in the worker-private staged configuration."""
    instrument_spooling_drivers(
        output, oracle_cpu_allowance_seconds=oracle_cpu_allowance_seconds
    )
    worker = output / "reference_worker.py"
    source = worker.read_text()
    if source.count(_SPOOL_IMPORT) != 1:
        raise ValueError("LayerNorm research worker staging seam changed")
    worker.write_text(source.replace(_SPOOL_IMPORT, _DEV_IMPORT))
    private = output / PRIVATE_FILE
    control = load_json_value(private)
    atomic_write_json_value(private, {**control, _SHIFT_FIELD: shift})
    private.chmod(0o600)
