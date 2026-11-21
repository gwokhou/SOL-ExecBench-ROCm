# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""P0/P1 tests for ReferenceService variation bounds, IPC ordering, and HMAC attestation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

import pytest
import torch
from sol_execbench_type_helpers import make_definition, make_workload

from sol_execbench.core import BenchmarkConfig
from sol_execbench.core.bench.performance_model.replay_evidence import (
    REPLAY_EVIDENCE_ITERATIONS,
    REPLAY_WARMUP_RUNS,
)
from sol_execbench.core.bench.reference_protocol import (
    TRUSTED_DEFINITION_FILE,
    ReferenceClient,
    ReferenceProtocolError,
    verify_reference_attestation,
)
from sol_execbench.core.bench.reference_service import ReferenceService
from sol_execbench.core.integrity.protocol_versions import WireProtocol


@contextmanager
def _pipe_pair() -> Iterator[tuple[Connection, Connection]]:
    read_fd, write_fd = os.pipe()
    sender = Connection(write_fd, readable=False, writable=True)
    receiver = Connection(read_fd, readable=True, writable=False)
    try:
        yield sender, receiver
    finally:
        sender.close()
        receiver.close()


def _stage_minimal_service(
    tmp_path: Path,
    *,
    warmup_runs: int = 0,
    iterations: int = 1,
    trials: int = 1,
) -> tuple[str, str]:
    definition = make_definition(
        name="vecadd_svc",
        op_type="elementwise",
        axes={"n": {"type": "const", "value": 8}},
        inputs={"x": {"shape": ["n"], "dtype": "float32"}},
        outputs={"z": {"shape": ["n"], "dtype": "float32"}},
        reference="def run(x):\n    return x + 1.0\n",
    )
    workload = make_workload(
        uuid="svc-wkl-0",
        axes={},
        inputs={"x": {"type": "random"}},
        checks=[{"type": "numeric", "output": "z"}],
    )
    config = BenchmarkConfig(
        lock_clocks=False,
        benchmark_reference=False,
        warmup_runs=warmup_runs,
        iterations=iterations,
        trials=trials,
    )
    (tmp_path / TRUSTED_DEFINITION_FILE).write_text(
        definition.model_dump_json()
    )
    (tmp_path / "workload.jsonl").write_text(workload.model_dump_json() + "\n")
    (tmp_path / "config.json").write_text(config.model_dump_json())
    return workload.uuid, "a1" * 32


def test_variation_index_allows_counter_and_scope_replay_with_small_config(
    tmp_path: Path,
) -> None:
    workload_uuid, nonce = _stage_minimal_service(
        tmp_path,
        warmup_runs=0,
        iterations=1,
        trials=1,
    )
    service = ReferenceService(
        tmp_path,
        device="cpu",
        input_nonce=nonce,
    )
    total_replay_iters = REPLAY_WARMUP_RUNS + REPLAY_EVIDENCE_ITERATIONS
    assert total_replay_iters >= 2

    for it in range(total_replay_iters):
        _, case, _, _ = service.handle(
            {
                "protocol": WireProtocol.REFERENCE_IPC,
                "operation": "timing_iteration",
                "workload_uuid": workload_uuid,
                "row_index": 0,
                "round_index": 9,
                "trial_index": 0,
                "iteration_index": it,
            }
        )
        service.validate_timing_outputs([case.inputs[0] + 1.0])


def test_validate_timing_outputs_encodes_payload_before_sending_control_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        _pipe_pair() as (req_writer, req_reader),
        _pipe_pair() as (_resp_writer, resp_reader),
    ):
        client = ReferenceClient(
            resp_reader,
            req_writer,
            token="secret",
            device="cpu",
        )
        # Force payload size failure via MAX_REFERENCE_PAYLOAD_BYTES = 1
        monkeypatch.setattr(
            "sol_execbench.core.bench.reference_protocol.MAX_REFERENCE_PAYLOAD_BYTES",
            1,
        )
        with pytest.raises(
            ReferenceProtocolError, match="payload is too large"
        ):
            client.validate_timing_outputs([torch.ones(16)])

        # Nothing was written to req_writer, so req_reader has no orphan control frame!
        assert req_reader.poll(0.0) is False

        # Also check invalid non-Tensor item in outputs raises before sending control frame
        with pytest.raises(ReferenceProtocolError, match="unsupported"):
            client.validate_timing_outputs(
                cast(list[torch.Tensor], ["not_a_tensor"])
            )
        assert req_reader.poll(0.0) is False


def test_reference_service_writes_hmac_verified_attestation_receipt(
    tmp_path: Path,
) -> None:
    workload_uuid, nonce = _stage_minimal_service(
        tmp_path,
        warmup_runs=1,
        iterations=2,
        trials=1,
    )
    key_hex = "b2" * 32
    receipt_path = tmp_path / "receipt.json"
    service = ReferenceService(
        tmp_path,
        device="cpu",
        input_nonce=nonce,
        attestation=(key_hex, receipt_path),
    )
    for round_index in range(10):
        service.handle(
            {
                "operation": "correctness",
                "workload_uuid": workload_uuid,
                "row_index": 0,
                "round_index": round_index,
            }
        )
    service.handle(
        {
            "operation": "timing",
            "workload_uuid": workload_uuid,
            "row_index": 0,
            "round_index": 9,
        }
    )
    for it in range(3):
        _, case, _, _ = service.handle(
            {
                "operation": "timing_iteration",
                "workload_uuid": workload_uuid,
                "row_index": 0,
                "round_index": 9,
                "trial_index": 0,
                "iteration_index": it,
            }
        )
        service.validate_timing_outputs([case.inputs[0] + 1.0])
    service.close()

    envelope = json.loads(receipt_path.read_text())
    payload = verify_reference_attestation(envelope, key_hex=key_hex)
    assert payload["pending_timing_validation"] is False
    wl = cast(dict[str, dict[str, object]], payload["workloads"])[workload_uuid]
    assert wl["all_correctness_rounds"] is True
    assert wl["correctness_rounds_completed"] == 10
    assert wl["timing_case_issued"] is True
    assert wl["timed_validations_passed"] == 3

    with pytest.raises(ReferenceProtocolError, match="signature mismatch"):
        verify_reference_attestation(envelope, key_hex="c3" * 32)
