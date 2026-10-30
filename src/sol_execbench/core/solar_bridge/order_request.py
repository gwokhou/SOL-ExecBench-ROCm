# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pinned original Python oracle for CPU-only contraction search workers."""

from pathlib import Path
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.analysis.order_contracts import MAX_ORDER_REPORT_BYTES, OrderContract
from solar.artifacts import sha256_bytes
from solar.artifacts.bundle import read_regular_file
from solar.artifacts.checksums import SHA256
from solar.artifacts.document import parse_yaml_artifact
from solar.contracts import ConversionRequest, InputFactory
from solar.order_contracts import OrderSearchRequest
from solar.schema_versions import SOLAR_ORDER_SCHEMA_VERSION

MAX_ORDER_ORACLE_BYTES = 1024 * 1024
ORACLE_REFERENCE_NAME = "reference"
ORACLE_INPUT_FACTORY_NAME = "input_factory"


class OrderOracle(BaseModel):
    """One reviewed local module containing both reference and input factory."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: Path
    sha256: SHA256
    reference_name: Annotated[str, Field(min_length=1, max_length=256)]
    trace_seed: Annotated[int, Field(strict=True)]


class OrderWorkerRequest(OrderContract):
    """Serializable process invocation; its source digest also pins oracle bytes."""

    search: OrderSearchRequest
    oracle: OrderOracle

    @model_validator(mode="after")
    def _same_oracle(self) -> OrderWorkerRequest:
        if self.oracle.sha256 != self.search.job.source.reference_sha256:
            raise ValueError(
                "oracle module must match the verified source digest"
            )
        return self


def load_order_request(path: Path) -> OrderWorkerRequest:
    """Read bounded bytes and reject the wrong version before interpreting fields."""
    payload = read_regular_file(path, max_bytes=MAX_ORDER_REPORT_BYTES)
    raw = parse_yaml_artifact(path, payload, max_bytes=MAX_ORDER_REPORT_BYTES)
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != SOLAR_ORDER_SCHEMA_VERSION
    ):
        raise ValueError("contraction search schema is not current")
    return OrderWorkerRequest.model_validate(raw.data)


def load_order_oracle(request: OrderWorkerRequest) -> ConversionRequest:
    """Execute exactly the pinned original bytes, only inside the isolated worker.

    This is an explicitly supplied Python program, not a pickle from the IR
    bundle. Its imports are subject to the caller's locked runtime environment.
    """
    payload = read_regular_file(
        request.oracle.path, max_bytes=MAX_ORDER_ORACLE_BYTES
    )
    if sha256_bytes(payload) != request.oracle.sha256:
        raise ValueError("original oracle module checksum differs")
    namespace: dict[str, object] = {
        "__name__": "solar_order_oracle",
        "__file__": str(request.oracle.path),
    }
    exec(compile(payload, str(request.oracle.path), "exec"), namespace)  # noqa: S102 -- explicitly pinned user-authored reference
    reference = namespace.get(ORACLE_REFERENCE_NAME)
    factory = namespace.get(ORACLE_INPUT_FACTORY_NAME)
    if not callable(reference) or not callable(factory):
        raise ValueError(
            "oracle module must define reference and input_factory callables"
        )
    return ConversionRequest(
        analysis_id="contraction-order-search",
        reference=reference,
        input_factory=cast(InputFactory, factory),
        reference_name=request.oracle.reference_name,
        reference_sha256=request.oracle.sha256,
        trace_seed=request.oracle.trace_seed,
        ir_path=request.search.job.source.ir_path,
        verification=request.search.job.source.verification,
    )
