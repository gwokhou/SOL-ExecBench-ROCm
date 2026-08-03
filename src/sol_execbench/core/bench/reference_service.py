# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trusted reference execution service for the staged evaluator."""

from __future__ import annotations

import os
import statistics
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, TextIO

import torch

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.correctness import check_output_shape_dtype
from sol_execbench.core.bench.eval_output_integrity import (
    stable_reference_outputs,
)
from sol_execbench.core.bench.eval_runtime import (
    load_reference_function,
    measure_reference_latency,
)
from sol_execbench.core.bench.io import (
    CustomInputFailureClass,
    CustomInputGenerationError,
    derive_custom_input_seed,
    gen_inputs,
    load_safetensors,
)
from sol_execbench.core.bench.output_checks import compare_output_checks
from sol_execbench.core.bench.performance_model.access_evidence import (
    summarize_integer_inputs,
)
from sol_execbench.core.bench.reference_protocol import (
    PROTOCOL_VERSION,
    TRUSTED_DEFINITION_FILE,
    ReferenceCase,
    ReferenceFailureKind,
    ReferenceProtocolError,
    receive_case,
    receive_json,
    send_case,
    send_failure,
    send_json,
)
from sol_execbench.core.bench.utils import call_and_collect_outputs
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.data.workload_validation import (
    validate_problem_contract,
)


class ReferenceRequestError(ValueError):
    """A candidate-side request did not satisfy the reference protocol."""


class InputGenerationError(RuntimeError):
    """Trusted input generation failed before the reference function ran."""

    def __init__(
        self,
        message: str,
        *,
        failure_class: CustomInputFailureClass,
    ) -> None:
        """Initialize a classified input-generation failure."""
        super().__init__(f"{failure_class}: {message}")
        self.failure_class = failure_class


class ReferenceService:
    """Own trusted reference code, input generation, output, and timing state."""

    def __init__(
        self,
        staging_dir: Path,
        *,
        device: str,
        input_nonce: str,
        definition_path: Path | None = None,
    ) -> None:
        """Initialize trusted reference execution for a staged problem."""
        try:
            nonce_bytes = bytes.fromhex(input_nonce)
        except ValueError as error:
            raise ValueError(
                "reference input nonce is not hexadecimal"
            ) from error
        if len(nonce_bytes) != 32:
            raise ValueError("reference input nonce must contain 32 bytes")
        self.staging_dir = staging_dir
        self.device = device
        self.input_nonce = input_nonce
        trusted_definition = (
            definition_path or staging_dir / TRUSTED_DEFINITION_FILE
        )
        self.definition = Definition.model_validate_json(
            trusted_definition.read_text(),
        )
        self.workloads = [
            Workload.model_validate_json(line)
            for line in (staging_dir / "workload.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        validate_problem_contract(self.definition, self.workloads)
        config_path = staging_dir / "config.json"
        self.config = (
            BenchmarkConfig.model_validate_json(config_path.read_text())
            if config_path.exists()
            else BenchmarkConfig()
        )
        self.reference_module, self.reference = load_reference_function(
            self.definition.reference,
        )
        self.custom_inputs = (
            getattr(
                self.reference_module,
                self.definition.custom_inputs_entrypoint,
            )
            if self.definition.custom_inputs_entrypoint
            else None
        )
        self.output_names = list(self.definition.outputs)
        self.output_dtypes = {
            name: dtype_str_to_torch_dtype(spec.dtype)
            for name, spec in self.definition.outputs.items()
        }
        self._safetensors: dict[str, dict[str, Any]] = {}
        self._pending_timing_validation: (
            tuple[
                Workload,
                list[Any],
                list[torch.Tensor],
            ]
            | None
        ) = None

    def handle(
        self,
        request: dict[str, Any],
    ) -> tuple[str, ReferenceCase, float, str | None]:
        """Validate one request and produce its trusted response."""
        operation, row_index, round_index, workload = self._request_context(
            request,
        )
        variation_index = self._variation_index(request, operation)
        self._require_validation_complete(operation)
        inputs = self._requested_inputs(
            workload,
            row_index,
            round_index,
            variation_index,
        )
        outputs = self._reference_outputs(workload, inputs)
        response_outputs = self._stage_timing_validation(
            operation,
            workload,
            inputs,
            outputs,
        )
        latency, failure = (
            self._timing(inputs) if operation == "timing" else (0.0, None)
        )
        return (
            operation,
            ReferenceCase(inputs=inputs, outputs=response_outputs),
            latency,
            failure,
        )

    def _request_context(
        self,
        request: dict[str, Any],
    ) -> tuple[str, int, int, Workload]:
        operation = request.get("operation")
        if not isinstance(operation, str) or operation not in {
            "correctness",
            "timing",
            "timing_iteration",
        }:
            raise ReferenceRequestError(
                f"unsupported reference operation: {operation!r}",
            )
        row_index = request.get("row_index")
        round_index = request.get("round_index")
        workload_uuid = request.get("workload_uuid")
        if not isinstance(row_index, int) or not 0 <= row_index < len(
            self.workloads,
        ):
            raise ReferenceRequestError("reference row_index is invalid")
        if not isinstance(round_index, int) or not 0 <= round_index < 10:
            raise ReferenceRequestError("reference round_index is invalid")
        workload = self.workloads[row_index]
        if workload_uuid != workload.uuid:
            raise ReferenceRequestError("reference workload identity mismatch")
        return operation, row_index, round_index, workload

    def _require_validation_complete(self, operation: str) -> None:
        if operation == "timing_iteration" and (
            self._pending_timing_validation is not None
        ):
            raise ReferenceRequestError(
                "previous timing iteration has not been validated",
            )

    def _requested_inputs(
        self,
        workload: Workload,
        row_index: int,
        round_index: int,
        variation_index: int | None,
    ) -> list[Any]:
        try:
            return self.prepare_inputs(
                workload,
                row_index,
                round_index,
                variation_index=variation_index,
            )
        except CustomInputGenerationError as exc:
            raise InputGenerationError(
                f"{exc}\n{exc.provenance.log_text()}",
                failure_class=exc.failure_class,
            ) from exc
        except Exception as exc:
            raise InputGenerationError(
                str(exc),
                failure_class=CustomInputFailureClass.ERROR,
            ) from exc

    def _reference_outputs(
        self,
        workload: Workload,
        inputs: list[Any],
    ) -> list[torch.Tensor]:
        resolved_axes = self.definition.get_resolved_axes_values(workload.axes)
        outputs = call_and_collect_outputs(
            self.reference,
            inputs,
            destination_passing_style=False,
            definition=self.definition,
            resolved_axes=resolved_axes,
            device=self.device,
            output_names=self.output_names,
            output_dtypes=self.output_dtypes,
        )
        return stable_reference_outputs(outputs, inputs)

    def _stage_timing_validation(
        self,
        operation: str,
        workload: Workload,
        inputs: list[Any],
        outputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if operation == "timing_iteration":
            self._pending_timing_validation = (workload, inputs, outputs)
            return []
        return outputs

    def validate_timing_outputs(self, actual: list[torch.Tensor]) -> None:
        """Validate one candidate result without disclosing the reference."""
        pending = self._pending_timing_validation
        if pending is None:
            raise ReferenceRequestError("no timing iteration awaits validation")
        self._pending_timing_validation = None
        workload, inputs, expected = pending
        issue = check_output_shape_dtype(expected, actual)
        if issue is not None:
            raise ReferenceRequestError(
                f"timed output shape or dtype is invalid: {issue}",
            )
        _, exceeds = compare_output_checks(
            self.definition,
            workload,
            inputs,
            expected,
            actual,
            0,
        )
        if exceeds:
            raise ReferenceRequestError(
                "timed output differs from the trusted reference",
            )

    def _variation_index(
        self,
        request: dict[str, Any],
        operation: str,
    ) -> int | None:
        if operation != "timing_iteration":
            return None
        trial_index = request.get("trial_index")
        iteration_index = request.get("iteration_index")
        if not isinstance(trial_index, int) or not 0 <= trial_index < max(
            self.config.trials,
            1,
        ):
            raise ReferenceRequestError("reference trial_index is invalid")
        iteration_count = self.config.warmup_runs + self.config.iterations
        if not isinstance(iteration_index, int) or not 0 <= iteration_index < (
            iteration_count
        ):
            raise ReferenceRequestError("reference iteration_index is invalid")
        return trial_index * iteration_count + iteration_index

    def prepare_inputs(
        self,
        workload: Workload,
        row_index: int,
        round_index: int,
        *,
        variation_index: int | None = None,
    ) -> list[Any]:
        """Generate one trusted, deterministically seeded workload input set."""
        safe_tensors = self._safetensors_for(workload)
        seed = derive_custom_input_seed(
            self.definition,
            workload,
            row_index=row_index,
            base_seed=self.config.seed,
            round_index=round_index,
            run_nonce=self.input_nonce,
            variation_index=variation_index,
        )
        return gen_inputs(
            self.definition,
            workload,
            device=self.device,
            safe_tensors=safe_tensors or None,
            custom_inputs_fn=self.custom_inputs,
            row_index=row_index,
            seed=seed,
        )

    def _safetensors_for(self, workload: Workload) -> dict[str, Any]:
        cached = self._safetensors.get(workload.uuid)
        if cached is not None:
            return cached
        if not any(
            value.type == "safetensors" for value in workload.inputs.values()
        ):
            result: dict[str, Any] = {}
        else:
            roots = [self.staging_dir]
            configured = os.environ.get("FLASHINFER_TRACE_DIR")
            if configured:
                roots.append(Path(configured))
            result = load_safetensors(self.definition, workload, roots)
        self._safetensors[workload.uuid] = result
        return result

    def _timing(self, inputs: list[Any]) -> tuple[float, str | None]:
        if not self.config.benchmark_reference:
            return 0.0, None
        results = [
            measure_reference_latency(
                self.reference,
                inputs,
                self.device,
                warmup=self.config.warmup_runs,
                rep=self.config.iterations,
                min_measurement_time_seconds=self.config.min_measurement_time_seconds,
            )
            for _ in range(self.config.trials)
        ]
        failure = next(
            (result.failure for result in results if result.failure),
            None,
        )
        if failure is not None:
            return 0.0, failure
        return statistics.mean(result.latency_ms for result in results), None


def _validated_request(
    request: dict[str, Any],
    *,
    token: str,
) -> dict[str, Any]:
    if request.get("protocol") != PROTOCOL_VERSION:
        raise ReferenceRequestError("reference protocol version mismatch")
    if request.get("token") != token:
        raise ReferenceRequestError("reference authentication failed")
    return request


def _serve_connection(
    reader: Connection,
    writer: Connection,
    service: ReferenceService,
    *,
    token: str,
) -> None:
    while True:
        try:
            request = _validated_request(receive_json(reader), token=token)
            if request.get("operation") == "shutdown":
                send_json(writer, {"ok": True, "protocol": PROTOCOL_VERSION})
                return
            if request.get("operation") == "timing_validation":
                actual = receive_case(reader, device=service.device)
                if actual.inputs:
                    raise ReferenceRequestError(
                        "timing validation payload must contain outputs only",
                    )
                service.validate_timing_outputs(actual.outputs)
                send_json(
                    writer,
                    {"ok": True, "protocol": PROTOCOL_VERSION},
                )
                continue
            operation, case, latency, failure = service.handle(request)
            access_patterns = (
                summarize_integer_inputs(
                    dict(
                        zip(
                            service.definition.inputs,
                            case.inputs,
                            strict=True,
                        )
                    )
                )
                if operation == "timing"
                else []
            )
            send_case(
                writer,
                case,
                reference_latency_ms=latency,
                timing_failure=failure,
                access_patterns=access_patterns,
            )
            del case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ReferenceRequestError as exc:
            send_failure(writer, str(exc))
        except InputGenerationError as exc:
            send_failure(
                writer,
                str(exc),
                kind=ReferenceFailureKind.INPUT_GENERATION,
            )
        except ReferenceProtocolError:
            return
        except (BrokenPipeError, EOFError):
            return
        except Exception as exc:  # noqa: BLE001 -- trusted request isolation
            try:
                send_failure(
                    writer,
                    f"trusted reference execution failed: {exc}",
                )
            except (BrokenPipeError, EOFError):
                return


def serve_reference_worker(
    staging_dir: str | Path,
    *,
    request_stream: Connection,
    response_stream: Connection,
    token: str,
    input_nonce: str,
    device: str,
    ready_stream: TextIO,
) -> None:
    """Serve one authenticated candidate worker over inherited private pipes."""
    service = ReferenceService(
        Path(staging_dir),
        device=device,
        input_nonce=input_nonce,
    )
    ready_stream.write("READY\n")
    ready_stream.flush()
    _serve_connection(request_stream, response_stream, service, token=token)


__all__ = [
    "InputGenerationError",
    "ReferenceRequestError",
    "ReferenceService",
    "serve_reference_worker",
]
