# SPDX-License-Identifier: Apache-2.0
"""Restricted-domain schedule completeness for the audited eager executor."""

from collections.abc import Sequence

from sol_execbench.core.bench.rocm_profiler.hip_sdk_memcpy import (
    DEFAULT_STREAM_ORDINAL,
)
from sol_execbench.core.bench.rocm_profiler.hip_sdk_models import (
    HIPSDKBoundDispatch,
    HIPSDKCopyDomain,
    HIPSDKScheduleQualification,
    HIPSDKScheduleReason,
)


def sdk_schedule_qualification(
    dispatches: Sequence[HIPSDKBoundDispatch],
    copy_domain: HIPSDKCopyDomain,
) -> HIPSDKScheduleQualification:
    """Complete only with lane-bound copies and no default-stream lane mixing.

    The default stream is the raw hipStream_t handle zero recorded by the
    audited collector; SDK ordinals never carry default-stream meaning.
    """
    lanes = {(dispatch.queue_id, dispatch.stream_id) for dispatch in dispatches}
    lanes |= {
        (copy.queue_id, copy.stream_ordinal)
        for copy in copy_domain.copies
        if copy.in_scope and copy.queue_id is not None
    }
    default_stream = any(
        dispatch.stream_handle == DEFAULT_STREAM_ORDINAL
        for dispatch in dispatches
    ) or any(
        copy.in_scope and copy.stream_handle == DEFAULT_STREAM_ORDINAL
        for copy in copy_domain.copies
    )
    reasons: set[HIPSDKScheduleReason] = set()
    if not copy_domain.qualified:
        reasons.add(HIPSDKScheduleReason.COPY_DOMAIN)
    if default_stream and len(lanes) > 1:
        reasons.add(HIPSDKScheduleReason.DEFAULT_STREAM)
    return HIPSDKScheduleQualification(
        schedule_complete=not reasons,
        default_stream_present=default_stream,
        lane_count=len(lanes),
        reason_codes=tuple(sorted(reasons, key=str)),
    )


__all__ = ["sdk_schedule_qualification"]
