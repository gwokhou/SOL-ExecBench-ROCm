from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sol_execbench.core.dataset import aka_probe_worker as worker
from sol_execbench.core.dataset.aka_compatibility import PROBE_RESULT_PREFIX


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "problem_dir": Path("."),
        "row_index": 0,
        "workload_uuid": "workload-1",
        "device": "cuda:0",
        "expected_arch": "gfx1200",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _payload(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    line = capsys.readouterr().out.strip()
    assert line.startswith(PROBE_RESULT_PREFIX)
    return json.loads(line[len(PROBE_RESULT_PREFIX) :])


def _service(*, workload_uuid: str = "workload-1") -> SimpleNamespace:
    workload = SimpleNamespace(uuid=workload_uuid, axes={})
    return SimpleNamespace(
        workloads=[workload],
        definition=SimpleNamespace(get_resolved_axes_values=lambda _axes: {}),
        prepare_inputs=lambda *_args: [torch.ones(2)],
        reference=lambda value: value + 1,
        output_names=["output"],
        output_dtypes={"output": torch.float32},
    )


def _matching_device(_device: str) -> SimpleNamespace:
    return SimpleNamespace(gfx_target="gfx1200")


def test_probe_reports_gpu_discovery_failure(capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        worker,
        "detect_rocm_device",
        lambda _device: (_ for _ in ()).throw(RuntimeError("no gpu")),
    )

    worker._run_probe(_args())

    assert _payload(capsys) == {
        "detail": "no gpu",
        "metrics": {},
        "reason_code": "gpu_unavailable",
        "status": "infrastructure_error",
    }


def test_probe_reports_target_and_workload_identity_mismatches(
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        worker,
        "detect_rocm_device",
        lambda _device: SimpleNamespace(gfx_target="gfx942"),
    )
    worker._run_probe(_args())
    assert _payload(capsys)["reason_code"] == "target_arch_mismatch"

    monkeypatch.setattr(worker, "detect_rocm_device", _matching_device)
    monkeypatch.setattr(
        worker,
        "ReferenceService",
        lambda *_args, **_kwargs: _service(),
    )
    worker._run_probe(_args(workload_uuid="other"))
    assert _payload(capsys)["reason_code"] == "reference_execution_failed"


def test_probe_rejects_oversized_input_and_case_payloads(
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setattr(worker, "detect_rocm_device", _matching_device)
    monkeypatch.setattr(
        worker,
        "ReferenceService",
        lambda *_args, **_kwargs: _service(),
    )
    monkeypatch.setattr(
        worker,
        "reference_values_storage_bytes",
        lambda _inputs: worker.MAX_REFERENCE_TENSOR_STORAGE_BYTES + 1,
    )

    worker._run_probe(_args())
    assert _payload(capsys)["reason_code"] == "reference_ipc_payload_limit"

    monkeypatch.setattr(
        worker,
        "reference_values_storage_bytes",
        lambda _inputs: 8,
    )
    monkeypatch.setattr(
        worker,
        "reference_case_storage_bytes",
        lambda _case: worker.MAX_REFERENCE_TENSOR_STORAGE_BYTES + 1,
    )
    monkeypatch.setattr(
        worker,
        "call_and_collect_outputs",
        lambda *_args, **_kwargs: [torch.ones(2)],
    )
    monkeypatch.setattr(
        worker,
        "stable_reference_outputs",
        lambda outputs, _inputs: outputs,
    )

    worker._run_probe(_args())
    assert _payload(capsys)["reason_code"] == "reference_ipc_payload_limit"


def test_probe_success_exercises_allocator_and_cache(
    capsys,
    monkeypatch,
) -> None:
    calls: list[str] = []
    fake_cache = SimpleNamespace(zero_=lambda: calls.append("zero"))
    monkeypatch.setattr(worker, "detect_rocm_device", _matching_device)
    monkeypatch.setattr(
        worker,
        "ReferenceService",
        lambda *_args, **_kwargs: _service(),
    )
    monkeypatch.setattr(
        worker,
        "reference_values_storage_bytes",
        lambda _inputs: 8,
    )
    monkeypatch.setattr(
        worker,
        "reference_case_storage_bytes",
        lambda _case: 16,
    )
    monkeypatch.setattr(
        worker,
        "call_and_collect_outputs",
        lambda *_args, **_kwargs: [torch.ones(2)],
    )
    monkeypatch.setattr(
        worker,
        "stable_reference_outputs",
        lambda outputs, _inputs: outputs,
    )
    monkeypatch.setattr(
        worker,
        "cache_clear_policy_for_device",
        lambda _device: SimpleNamespace(clear_buffer_bytes=64),
    )
    monkeypatch.setattr(
        worker,
        "ShiftingMemoryPoolAllocator",
        lambda *_args: SimpleNamespace(
            get_unique_args=lambda: calls.append("allocate"),
        ),
    )
    monkeypatch.setattr(
        worker.torch,
        "empty",
        lambda *_args, **_kwargs: fake_cache,
    )
    monkeypatch.setattr(
        worker.torch.cuda,
        "synchronize",
        lambda _device: calls.append("sync"),
    )

    worker._run_probe(_args())

    payload = _payload(capsys)
    assert payload["status"] == "compatible"
    assert payload["metrics"] == {
        "cache_clear_bytes": 64,
        "input_storage_bytes": 8,
        "reference_case_bytes": 16,
    }
    assert calls == ["allocate", "zero", "sync"]


@pytest.mark.parametrize(
    ("exception", "reason_code"),
    [
        (torch.cuda.OutOfMemoryError("oom"), "probe_oom"),
        (RuntimeError("bad reference"), "reference_execution_failed"),
    ],
)
def test_probe_classifies_execution_failures(
    exception: Exception,
    reason_code: str,
    capsys,
    monkeypatch,
) -> None:
    service = _service()
    service.prepare_inputs = lambda *_args: (_ for _ in ()).throw(exception)
    monkeypatch.setattr(worker, "detect_rocm_device", _matching_device)
    monkeypatch.setattr(
        worker,
        "ReferenceService",
        lambda *_args, **_kwargs: service,
    )

    worker._run_probe(_args())

    assert _payload(capsys)["reason_code"] == reason_code


def test_probe_main_parses_closed_cli_contract(monkeypatch, tmp_path) -> None:
    captured: list[argparse.Namespace] = []
    monkeypatch.setattr(worker, "_run_probe", captured.append)
    monkeypatch.setattr(
        worker.argparse.ArgumentParser,
        "parse_args",
        lambda _parser: _args(problem_dir=tmp_path),
    )

    worker.main()

    assert captured[0].problem_dir == tmp_path
