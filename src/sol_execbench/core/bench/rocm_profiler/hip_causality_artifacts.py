# SPDX-License-Identifier: Apache-2.0
"""Bounded host-order correlation from a single ROCprofiler JSON artifact."""

import json
from pathlib import Path

from pydantic import ValidationError

from sol_execbench.core.bench.rocm_profiler.hip_causality import (
    HIPCausalTrace,
    correlate_hip_events,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_json import (
    SDKCausalDocument,
    causal_trace,
)
from sol_execbench.core.bench.rocm_profiler.hip_causality_models import (
    RECORD_APIS,
    WAIT_API,
    HIPEventCausality,
    HIPLinkReason,
)
from sol_execbench.core.bench.rocm_profiler.hip_event_artifacts import (
    MAX_JSON_BYTES,
)
from sol_execbench.core.integrity.checksums import sha256_bytes


def read_hip_event_causality(
    path: Path,
) -> tuple[list[HIPEventCausality], list[str]]:
    """Correlate JSON only; rocpd lacks the required success return values."""
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_JSON_BYTES + 1)
        if len(content) > MAX_JSON_BYTES:
            return [], [f"{path.name}: {HIPLinkReason.LIMIT}"]
        payload = json.loads(content)
        if (
            not isinstance(payload, dict)
            or "rocprofiler-sdk-tool" not in payload
        ):
            return [], []
        document = SDKCausalDocument.model_validate(payload)
        digest = sha256_bytes(content)
        results = []
        for index, capture in enumerate(document.captures):
            if not capture.buffer_records.hip_event and not any(
                capture.operation(call, "HIP_RUNTIME_API_EXT")
                in RECORD_APIS | {WAIT_API}
                for call in capture.buffer_records.hip_api
            ):
                continue
            trace = causal_trace(capture, f"{path.name}:{index}", digest)
            if _unsupported_lifetime(trace):
                return [], [f"{path.name}: {HIPLinkReason.UNSUPPORTED}"]
            results.append(correlate_hip_events(trace))
        return results, []
    except OSError, ValueError, IndexError, RecursionError, ValidationError:
        return [], [f"{path.name}: {HIPLinkReason.INVALID}"]


def _unsupported_lifetime(trace: HIPCausalTrace) -> bool:
    last_wait = max(
        (call.end_ns for call in trace.calls if call.name == WAIT_API),
        default=0,
    )
    return any(
        "Graph" in call.name
        or "Capture" in call.name
        or call.name == "hipIpcOpenEventHandle"
        or call.name.endswith("_spt")
        or (
            call.name in {"hipStreamDestroy", "hipDeviceReset"}
            and call.start_ns <= last_wait
        )
        for call in trace.calls
    )
