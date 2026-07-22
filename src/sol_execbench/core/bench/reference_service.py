# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Trusted reference execution service for the staged evaluator."""

from __future__ import annotations

import json
import os
import statistics
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, TextIO

import torch

from sol_execbench.core.bench.config import BenchmarkConfig
from sol_execbench.core.bench.eval_output_integrity import stable_reference_outputs
from sol_execbench.core.bench.eval_runtime import (
    load_reference_function,
    measure_reference_latency,
)
from sol_execbench.core.bench.io import (
    GEN_INPUTS_ERROR,
    CustomInputGenerationError,
    derive_custom_input_seed,
    gen_inputs,
    load_safetensors,
)
from sol_execbench.core.bench.reference_protocol import (
    PROTOCOL_VERSION,
    ReferenceCase,
    ReferenceFailureKind,
    ReferenceProtocolError,
    TRUSTED_DEFINITION_FILE,
    receive_json,
    send_case,
    send_failure,
    send_json,
)
from sol_execbench.core.bench.utils import call_and_collect_outputs
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype
from sol_execbench.core.data.workload import Workload


class ReferenceRequestError(ValueError):
    """A candidate-side request did not satisfy the reference protocol."""


class InputGenerationFailure(RuntimeError):
    """Trusted input generation failed before the reference function ran."""

    def __init__(self, message: str, *, failure_class: str) -> None:
        super().__init__(f"{failure_class}: {message}")
        self.failure_class = failure_class


class ReferenceService:
    """Own trusted reference code, input generation, output, and timing state."""

    def __init__(
        self,
        staging_dir: Path,
        *,
        device: str,
        definition_path: Path | None = None,
    ) -> None:
        self.staging_dir = staging_dir
        self.device = device
        trusted_definition = definition_path or staging_dir / TRUSTED_DEFINITION_FILE
        self.definition = Definition.model_validate_json(trusted_definition.read_text())
        self.workloads = [
            Workload.model_validate_json(line)
            for line in (staging_dir / "workload.jsonl").read_text().splitlines()
            if line.strip()
        ]
        config_path = staging_dir / "config.json"
        self.config = (
            BenchmarkConfig(**json.loads(config_path.read_text()))
            if config_path.exists()
            else BenchmarkConfig()
        )
        self.reference_module, self.reference = load_reference_function(
            self.definition.reference
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

    def handle(
        self, request: dict[str, Any]
    ) -> tuple[str, ReferenceCase, float, str | None]:
        """Validate one request and produce its trusted response."""
        operation = request.get("operation")
        if operation not in {"correctness", "timing"}:
            raise ReferenceRequestError(
                f"unsupported reference operation: {operation!r}"
            )
        row_index = request.get("row_index")
        round_index = request.get("round_index")
        workload_uuid = request.get("workload_uuid")
        if not isinstance(row_index, int) or not 0 <= row_index < len(self.workloads):
            raise ReferenceRequestError("reference row_index is invalid")
        if not isinstance(round_index, int) or not 0 <= round_index < 10:
            raise ReferenceRequestError("reference round_index is invalid")
        workload = self.workloads[row_index]
        if workload_uuid != workload.uuid:
            raise ReferenceRequestError("reference workload identity mismatch")
        try:
            inputs = self.prepare_inputs(workload, row_index, round_index)
        except CustomInputGenerationError as exc:
            raise InputGenerationFailure(
                f"{exc}\n{exc.provenance.log_text()}",
                failure_class=exc.failure_class,
            ) from exc
        except Exception as exc:
            raise InputGenerationFailure(
                str(exc), failure_class=GEN_INPUTS_ERROR
            ) from exc
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
        outputs = stable_reference_outputs(outputs, inputs)
        latency, failure = (
            self._timing(inputs) if operation == "timing" else (0.0, None)
        )
        return (
            operation,
            ReferenceCase(inputs=inputs, outputs=outputs),
            latency,
            failure,
        )

    def prepare_inputs(
        self, workload: Workload, row_index: int, round_index: int
    ) -> list[Any]:
        """Generate one trusted, deterministically seeded workload input set."""

        safe_tensors = self._safetensors_for(workload)
        seed = derive_custom_input_seed(
            self.definition,
            workload,
            row_index=row_index,
            base_seed=self.config.seed,
            round_index=round_index,
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
        if not any(value.type == "safetensors" for value in workload.inputs.values()):
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
        failure = next((result.failure for result in results if result.failure), None)
        if failure is not None:
            return 0.0, failure
        return statistics.mean(result.latency_ms for result in results), None


def _validated_request(request: dict[str, Any], *, token: str) -> dict[str, Any]:
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
            _, case, latency, failure = service.handle(request)
            send_case(
                writer,
                case,
                reference_latency_ms=latency,
                timing_failure=failure,
            )
            del case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ReferenceRequestError as exc:
            send_failure(writer, str(exc))
        except InputGenerationFailure as exc:
            send_failure(
                writer,
                str(exc),
                kind=ReferenceFailureKind.INPUT_GENERATION,
            )
        except ReferenceProtocolError:
            return
        except (BrokenPipeError, EOFError):
            return
        except Exception as exc:
            try:
                send_failure(writer, f"trusted reference execution failed: {exc}")
            except (BrokenPipeError, EOFError):
                return


def serve_reference_worker(
    staging_dir: str | Path,
    *,
    request_stream: Connection,
    response_stream: Connection,
    token: str,
    device: str,
    ready_stream: TextIO,
) -> None:
    """Serve one authenticated candidate worker over inherited private pipes."""
    service = ReferenceService(Path(staging_dir), device=device)
    ready_stream.write("READY\n")
    ready_stream.flush()
    _serve_connection(request_stream, response_stream, service, token=token)


__all__ = [
    "InputGenerationFailure",
    "ReferenceRequestError",
    "ReferenceService",
    "serve_reference_worker",
]
