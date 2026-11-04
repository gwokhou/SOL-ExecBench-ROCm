# SPDX-License-Identifier: Apache-2.0
"""Strict clock policy and persistent archive for production scope replay."""

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from sol_execbench.cli.evaluation.profile_mode import ProfileMode
from sol_execbench.cli.evaluation.requests import EvaluationRequest
from sol_execbench.cli.protocol import CliFailure
from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    SCOPE_ARCHIVE_SUFFIX,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    ROCR_UUID_PATTERN,
)
from sol_execbench.core.bench.rocm_profiler.pmc_clock import pmc_clock_window
from sol_execbench.core.process.environment import ENV_ROCR_VISIBLE_DEVICES


@contextmanager
def scope_clock_window(request: EvaluationRequest) -> Iterator[Path | None]:
    """Require one physical GPU and STABLE_PEAK for canonical run and replay."""
    if request.profile is not ProfileMode.ROCPROFV3_SCOPE:
        yield None
        return
    _require_scope_inputs(request)
    archive = _scope_archive(request)
    with pmc_clock_window(archive) as lease:
        if not lease.locked:
            raise CliFailure(
                "Scope replay requires verified STABLE_PEAK",
                code="scope_clock_unverified",
            )
        yield archive


def _require_scope_inputs(request: EvaluationRequest) -> None:
    if request.profile is not ProfileMode.ROCPROFV3_SCOPE:
        return
    if request.output_file is None:
        raise CliFailure(
            "rocprofv3-scope requires --trace-output",
            code="scope_output_required",
        )
    if (
        re.fullmatch(
            ROCR_UUID_PATTERN, os.environ.get(ENV_ROCR_VISIBLE_DEVICES, "")
        )
        is None
    ):
        raise CliFailure(
            "rocprofv3-scope requires one physical ROCR_VISIBLE_DEVICES UUID",
            code="scope_device_uuid_required",
        )


def _scope_archive(request: EvaluationRequest) -> Path:
    if request.output_file is None:
        raise ValueError("scope_output_required")
    root = request.output_file.resolve().with_name(
        request.output_file.name + SCOPE_ARCHIVE_SUFFIX
    )
    directory = root / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    return directory
