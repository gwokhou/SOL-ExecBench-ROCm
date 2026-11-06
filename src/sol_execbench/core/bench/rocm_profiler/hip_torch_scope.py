# SPDX-License-Identifier: Apache-2.0
"""Qualify an audited eager PyTorch scope without hiding implicit dispatches."""

from pathlib import Path
from uuid import UUID

from sol_execbench.core.bench.rocm_profiler.hip_causality_json import SDKCapture
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
    HIPScopeCaptureView,
    HIPScopeError,
    read_bound_json,
    read_scope_capture,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    HIPScopeReason,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_coverage import (
    bind_dispatches,
    validate_lifetime,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_links import sdk_event_links
from sol_execbench.core.bench.rocm_profiler.hip_sdk_memcpy import (
    memcpy_copy_domain,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    MAX_TORCH_SCOPE_BYTES,
    SDKAPI,
    HIPSDKCopyDomain,
    HIPSDKEngineCopy,
    HIPSDKLedger,
    HIPTorchExpectation,
    HIPTorchReceipt,
    HIPTorchScope,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_schedule import (
    sdk_schedule_qualification,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_validation import (
    require_scope,
    validate_apis,
    validate_boundaries,
    validate_dispatches,
    validate_events,
)
from sol_execbench.core.integrity.checksums import sha256_file

TORCH_SCOPE_ARTIFACTS = frozenset(
    {"torch_results.json", "ledger.json", "workload.json"}
)


def qualify_torch_scope(
    directory: Path,
    expected: HIPTorchExpectation,
    artifact_sha256: dict[str, str],
    *,
    executor_path: Path,
    collector_path: Path,
    definition_path: Path,
) -> HIPTorchScope:
    """Require retained digests and an audited executor; never infer trust."""
    try:
        return _qualify_torch_scope(
            directory,
            expected,
            artifact_sha256,
            executor_path=executor_path,
            collector_path=collector_path,
            definition_path=definition_path,
        )
    except HIPScopeError:
        raise
    except (ValueError, KeyError, IndexError, OSError) as error:
        raise HIPScopeError(HIPScopeReason.INVALID) from error


def _qualify_torch_scope(
    directory: Path,
    expected: HIPTorchExpectation,
    artifact_sha256: dict[str, str],
    *,
    executor_path: Path,
    collector_path: Path,
    definition_path: Path,
) -> HIPTorchScope:
    """Require retained digests and an audited executor; never infer trust."""
    require_scope(
        set(artifact_sha256) == TORCH_SCOPE_ARTIFACTS, HIPScopeReason.INVALID
    )
    payloads = {
        name: read_bound_json(
            directory / name, digest, max_bytes=MAX_TORCH_SCOPE_BYTES
        )
        for name, digest in artifact_sha256.items()
    }
    ledger = HIPSDKLedger.model_validate_json(payloads["ledger.json"])
    capture, view = read_scope_capture(payloads["torch_results.json"])
    receipt = HIPTorchReceipt.model_validate_json(payloads["workload.json"])
    for path, digest in (
        (executor_path, expected.executor_sha256),
        (collector_path, expected.collector_sha256),
        (definition_path, expected.definition_sha256),
    ):
        require_scope(sha256_file(path) == digest, HIPScopeReason.IDENTITY)
    agent = _identity(capture, view, ledger, receipt, expected)
    begin, end, seal = validate_boundaries(ledger, capture)
    validate_apis(ledger, capture, seal)
    enqueues = validate_dispatches(ledger, capture, agent)
    validate_events(ledger, capture, seal)
    validate_lifetime(capture, ledger, begin, end, seal)
    copy_domain = memcpy_copy_domain(ledger, capture, begin, end)
    dispatches, implicit = bind_dispatches(
        capture, view, enqueues, ledger_apis=_ledger_apis(ledger)
    )
    links = sdk_event_links(ledger, capture, enqueues, _lane_bound(copy_domain))
    schedule = sdk_schedule_qualification(dispatches, copy_domain)
    return HIPTorchScope(
        scope=expected,
        artifact_sha256=artifact_sha256,
        process_id=ledger.pid,
        dispatches=dispatches,
        outside_dispatch_ids=[r.dispatch for r in enqueues if r.scope == 0],
        implicit_dispatch_ids=implicit,
        links=links,
        copy_domain=copy_domain,
        schedule=schedule,
        schedule_complete=schedule.schedule_complete,
    )


def _ledger_apis(ledger: HIPSDKLedger) -> dict[int, SDKAPI]:
    """Ledger API rows keyed by correlation for stream-handle lookups."""
    return {r.corr: r for r in ledger.records if isinstance(r, SDKAPI)}


def _lane_bound(copy_domain: HIPSDKCopyDomain) -> list[HIPSDKEngineCopy]:
    """Only lane-bound in-scope copies may serve as event link endpoints."""
    return [
        copy
        for copy in copy_domain.copies
        if copy.in_scope and copy.queue_id is not None
    ]


def _identity(
    capture: SDKCapture,
    view: HIPScopeCaptureView,
    ledger: HIPSDKLedger,
    receipt: HIPTorchReceipt,
    expected: HIPTorchExpectation,
) -> int:
    require_scope(
        ledger.run_id == expected.run_id
        and ledger.pid == capture.metadata.pid == receipt.process_id
        and tuple(capture.metadata.command) == expected.application_command
        and all(
            getattr(receipt, key) == getattr(expected, key)
            for key in (
                "run_id",
                "workload_id",
                "definition_sha256",
                "executor_sha256",
                "collector_sha256",
            )
        ),
        HIPScopeReason.IDENTITY,
    )
    # HIP returns 16 ASCII hex characters through hipUUID; Torch prints them
    # as a standard UUID. SDK uses the little-endian integer encoding instead.
    hip_uuid = "GPU-" + UUID(receipt.device_uuid).bytes.decode("ascii")
    require_scope(hip_uuid == expected.device_uuid, HIPScopeReason.IDENTITY)
    agents = [
        a.id.handle for a in view.agents if a.uuid.hex_value() == hip_uuid
    ]
    require_scope(len(agents) == 1, HIPScopeReason.IDENTITY)
    return agents[0]
