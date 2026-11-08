"""Worker-private research observation, sharing the incumbent reference service."""

import base64
import json
import math
import resource
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load

from scripts.internal.solar.e2_contracts import (
    MAX_OBSERVATION_BYTES,
    MAX_SESSION_OBSERVATIONS,
    OBSERVATION_OPERATION,
    PRIVATE_FILE,
    Mode,
)
from scripts.internal.solar.e2_observation import (
    Observation,
    diagnose,
    encode_observation,
)
from sol_execbench.core.bench.custom_inputs import derive_custom_input_seed
from sol_execbench.core.bench.judge.contracts import (
    JudgeCase,
    JudgeFamily,
    JudgeInstance,
    JudgeStrategy,
    JudgeTask,
)
from sol_execbench.core.bench.judge.inputs import (
    JudgeInputs,
    generate_seeded_inputs,
)
from sol_execbench.core.bench.judge.interval_oracle import OracleUncertainError
from sol_execbench.core.bench.judge.storage import tensor_identity
from sol_execbench.core.bench.reference_protocol import ReferenceCase
from sol_execbench.core.bench.reference_service import (
    ReferenceRequestError,
    ReferenceService,
    serve_reference_worker,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_bytes


class ObservationService(ReferenceService):
    """Never import candidate code, send oracle data, or add candidate invocations."""

    def __init__(self, staging_dir: Path, **kwargs: Any) -> None:
        """Consume private session state before the orchestrator starts the candidate."""
        super().__init__(staging_dir, **kwargs)
        private_path = staging_dir / PRIVATE_FILE
        private = json.loads(private_path.read_text())
        private_path.unlink()
        self.case = JudgeCase(
            task=JudgeTask(private["task"]),
            family=JudgeFamily(private["family"]),
            instance=JudgeInstance(private["instance"]),
            workload_index=private["workload_index"],
        )
        self.strategy = JudgeStrategy(private["strategy"])
        self.mode = Mode(private["mode"])
        self.config = self.config.model_copy(
            update={"seed": private["base_seed"]}
        )
        self.rows: list[dict[str, object]] = []
        self.buffers: list[bytes] = []
        self.pending_correctness: (
            tuple[int, int, JudgeInputs, torch.Tensor] | None
        ) = None
        self.latest_inputs: JudgeInputs | None = None
        self.latest_coordinate: dict[str, object] = {}
        self.stop_reason: str | None = None
        self.service_seconds = 0.0
        self.current_checks_seconds = 0.0
        self.reference_calls = 0
        self._original_reference = self.reference
        self.reference = self._counted_reference

    def _counted_reference(self, *args: Any, **kwargs: Any) -> Any:
        self.reference_calls += 1
        return self._original_reference(*args, **kwargs)

    def prepare_inputs(
        self,
        workload: Workload,
        row_index: int,
        round_index: int,
        *,
        variation_index: int | None = None,
    ) -> list[Any]:
        """Retain actual inputs and verify the matching frozen witness metadata."""
        inputs = super().prepare_inputs(
            workload, row_index, round_index, variation_index=variation_index
        )
        seed = derive_custom_input_seed(
            self.definition,
            workload,
            row_index=row_index,
            base_seed=self.config.seed,
            round_index=round_index,
            run_nonce=self.input_nonce,
            variation_index=variation_index,
        )
        generated = generate_seeded_inputs(
            self.case.task,
            self.case.family,
            self.definition.get_resolved_axes_values(workload.axes),
            seed,
        )
        actual = tuple(
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=torch.float32)
            for value in inputs
        )
        if tensor_identity(actual) != tensor_identity(generated.values):
            raise ValueError(
                "actual trusted inputs differ from the frozen generator"
            )
        self.latest_inputs = JudgeInputs(
            values=actual,
            witness_columns=generated.witness_columns,
            construction=generated.construction,
        )
        return inputs

    def handle(
        self, request: dict[str, Any]
    ) -> tuple[str, ReferenceCase, float, str | None]:
        """Preserve incumbent requests and require exact correctness observation order."""
        if request.get("operation") == OBSERVATION_OPERATION:
            return self._observe_correctness(request)
        if self.pending_correctness is not None:
            raise ReferenceRequestError(
                "previous correctness output was not observed"
            )
        start = time.monotonic()
        result = super().handle(request)
        self.service_seconds += time.monotonic() - start
        operation, case, _, _ = result
        self.latest_coordinate = {
            key: request[key]
            for key in (
                "operation",
                "row_index",
                "round_index",
                "trial_index",
                "iteration_index",
            )
            if key in request
        }
        if operation == "correctness":
            if self.latest_inputs is None:
                raise RuntimeError("trusted inputs not retained")
            self.pending_correctness = (
                request["row_index"],
                request["round_index"],
                self.latest_inputs,
                case.outputs[0].detach().cpu().clone(),
            )
        return result

    def _observe_correctness(
        self, request: dict[str, Any]
    ) -> tuple[str, ReferenceCase, float, None]:
        pending = self.pending_correctness
        if (
            pending is None
            or (request.get("row_index"), request.get("round_index"))
            != pending[:2]
        ):
            raise ReferenceRequestError(
                "judge correctness observation identity mismatch"
            )
        payload = request.get("payload")
        seconds = request.get("checks_seconds")
        if not isinstance(payload, str) or len(payload) > 4 * (
            (MAX_OBSERVATION_BYTES + 2) // 3
        ):
            raise ReferenceRequestError(
                "judge observation payload length invalid"
            )
        if (
            not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ReferenceRequestError("judge check duration invalid")
        decoded = base64.b64decode(payload, validate=True)
        tensors = load(decoded)
        if len(decoded) > MAX_OBSERVATION_BYTES or set(tensors) != {"output"}:
            raise ReferenceRequestError("judge observation payload invalid")
        numeric = request.get("numeric_passed")
        if numeric is not None and not isinstance(numeric, bool):
            raise ReferenceRequestError("judge numeric result is invalid")
        self.latest_coordinate["incumbent_numeric_passed"] = numeric
        self.pending_correctness = None
        self.current_checks_seconds += seconds
        self._record(
            Observation(
                inputs=pending[2],
                reference=pending[3],
                actual=tensors["output"],
                coordinate=dict(self.latest_coordinate),
            )
        )
        return (
            OBSERVATION_OPERATION,
            ReferenceCase(inputs=[], outputs=[]),
            0.0,
            None,
        )

    def validate_timing_outputs(self, actual: list[torch.Tensor]) -> None:
        """Keep incumbent timed-output validation and retain even rejected outputs."""
        pending = self._pending_timing_validation
        if pending is None or self.latest_inputs is None or len(actual) != 1:
            raise ReferenceRequestError(
                "judge timed observation has no matching reference"
            )
        observation = Observation(
            inputs=self.latest_inputs,
            reference=pending[2][0].detach().cpu().clone(),
            actual=actual[0].detach().cpu().clone(),
            coordinate=dict(self.latest_coordinate),
        )
        start = time.monotonic()
        passed = False
        try:
            super().validate_timing_outputs(actual)
            passed = True
        finally:
            self.current_checks_seconds += time.monotonic() - start
            observation.coordinate["incumbent_timing_validation_passed"] = (
                passed
            )
            self._record(observation)

    def _record(self, observation: Observation) -> None:
        if len(self.rows) >= MAX_SESSION_OBSERVATIONS:
            raise ReferenceRequestError("judge actual-call budget exceeded")
        extra = {}
        diagnostic: dict[str, object] = {
            "diagnostic_status": "deferred_incumbent_mode"
        }
        if self.mode is Mode.DIAGNOSTIC and self.stop_reason is None:
            try:
                diagnostic, extra = diagnose(
                    self.case,
                    self.strategy,
                    observation,
                    self.definition,
                    self.workloads[0],
                )
            except (OracleUncertainError, TimeoutError, ValueError) as exc:
                self.stop_reason = f"{type(exc).__name__}: {exc}"
                diagnostic = {
                    "diagnostic_status": "oracle_uncertain",
                    "message": self.stop_reason,
                }
        encode_start = time.monotonic()
        record, payload = encode_observation(observation, extra)
        record["evidence_encoding_wall_seconds"] = (
            time.monotonic() - encode_start
        )
        self.rows.append(
            {**record, **diagnostic, "artifact_sha256": sha256_bytes(payload)}
        )
        self.buffers.append(payload)

    def close(self) -> None:
        """Publish worker evidence only after candidate requests cease."""
        output = self.staging_dir / "judge-observations"
        output.mkdir(exist_ok=False)
        for index, (record, payload) in enumerate(
            zip(self.rows, self.buffers, strict=True)
        ):
            name = f"call-{index:03d}.npz"
            (output / name).write_bytes(payload)
            record["artifact"] = name
        atomic_write_json_value(
            output / "observations.json",
            {
                "research_format": "E2-private-observations-v1",
                "performance_authority": False,
                "private_nonce_after_channel_close": self.input_nonce,
                "observations": self.rows,
                "stop_reason": self.stop_reason,
                "pending_correctness": self.pending_correctness is not None,
                "reference_calls": self.reference_calls,
                "reference_prepare_service_wall_seconds": self.service_seconds,
                "current_checks_seconds": self.current_checks_seconds,
                "reference_worker_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            },
        )


def serve_observed_worker(staging_dir: str | Path, **kwargs: Any) -> None:
    """Activate observation only in the explicitly staged E2 research driver."""
    serve_reference_worker(
        staging_dir, service_type=ObservationService, **kwargs
    )
