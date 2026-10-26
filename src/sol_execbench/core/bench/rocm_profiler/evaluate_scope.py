# SPDX-License-Identifier: Apache-2.0
"""Fail-closed qualification using parent-owned intent and SDK channel agreement."""

from pathlib import Path
from uuid import UUID

from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    SCOPE_ARTIFACTS,
    SCOPE_LEDGER_FILE,
    SCOPE_RECEIPT_FILE,
    SCOPE_TRACE_FILE,
    EvaluateScopeIntent,
    EvaluateScopeReceipt,
    EvaluateScopeResult,
    EvaluateScopeStatus,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_artifacts import (
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
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    MAX_TORCH_SCOPE_BYTES,
    HIPSDKLedger,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_validation import (
    require_scope,
    validate_apis,
    validate_boundaries,
    validate_dispatches,
    validate_events,
)
from sol_execbench.core.bench.rocm_profiler.scope_sources import (
    scope_runtime_digests,
)
from sol_execbench.core.integrity.checksums import (
    sha256_file,
    stable_json_checksum,
)


def verify_staged_scope(staging: Path, expected: EvaluateScopeIntent) -> None:
    """Reject source drift; expected hashes come from memory, not a child file."""
    require_scope(
        scope_runtime_digests() == expected.runtime_sha256,
        HIPScopeReason.IDENTITY,
    )
    for relative, digest in expected.staged_sha256.items():
        path = staging / relative
        require_scope(
            path.resolve().is_relative_to(staging.resolve())
            and not path.is_symlink()
            and sha256_file(path) == digest,
            HIPScopeReason.IDENTITY,
        )


def qualify_evaluate_scope(
    directory: Path,
    expected: EvaluateScopeIntent,
    hashes: dict[str, str],
    *,
    staging: Path,
    collector: Path,
    returncode: int,
) -> EvaluateScopeResult:
    """Do not accept summaries or child-provided expected identities."""
    try:
        require_scope(
            returncode == 0 and set(hashes) == SCOPE_ARTIFACTS,
            HIPScopeReason.INVALID,
        )
        verify_staged_scope(staging, expected)
        require_scope(
            sha256_file(collector) == expected.collector_sha256,
            HIPScopeReason.IDENTITY,
        )
        payloads = {
            name: read_bound_json(
                directory / name, digest, max_bytes=MAX_TORCH_SCOPE_BYTES
            )
            for name, digest in hashes.items()
        }
        ledger = HIPSDKLedger.model_validate_json(payloads[SCOPE_LEDGER_FILE])
        receipt = EvaluateScopeReceipt.model_validate_json(
            payloads[SCOPE_RECEIPT_FILE]
        )
        capture, view = read_scope_capture(payloads[SCOPE_TRACE_FILE])
        uuid = "GPU-" + UUID(receipt.device_uuid).bytes.decode("ascii")
        require_scope(
            receipt.intent_sha256
            == stable_json_checksum(expected.model_dump(mode="json"))
            and ledger.run_id == expected.run_id
            and ledger.pid == receipt.process_id == capture.metadata.pid
            and tuple(capture.metadata.command) == expected.application_command
            and uuid == expected.device_uuid,
            HIPScopeReason.IDENTITY,
        )
        agents = [
            a.id.handle for a in view.agents if a.uuid.hex_value() == uuid
        ]
        require_scope(len(agents) == 1, HIPScopeReason.IDENTITY)
        begin, end, seal = validate_boundaries(ledger, capture)
        validate_apis(ledger, capture, seal)
        enqueues = validate_dispatches(ledger, capture, agents[0])
        validate_events(ledger, capture)
        validate_lifetime(capture, ledger, begin, end, seal)
        dispatches, implicit = bind_dispatches(capture, view, enqueues)
        links = sdk_event_links(ledger, capture, enqueues)
        return EvaluateScopeResult(
            intent=expected,
            status=EvaluateScopeStatus.QUALIFIED,
            artifact_sha256=hashes,
            dispatches=dispatches,
            implicit_dispatch_ids=implicit,
            links=links,
            outside_dispatch_ids=[r.dispatch for r in enqueues if r.scope == 0],
        )
    except HIPScopeError:
        raise
    except (OSError, ValueError, KeyError, IndexError) as error:
        raise HIPScopeError(HIPScopeReason.INVALID) from error
