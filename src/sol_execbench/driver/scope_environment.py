# SPDX-License-Identifier: Apache-2.0
"""Keep reference-worker activity outside candidate-only SDK collection."""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

from sol_execbench.core.bench.rocm_profiler.evaluate_scope_models import (
    SCOPE_ENV,
    SCOPE_LAUNCH_FILE,
)

PROFILER_ENV_PREFIXES = ("ROCP", "SOL_SCOPE_")
PROFILER_LOADER_NAMES = frozenset({"LD_PRELOAD", "LD_AUDIT", SCOPE_ENV})


def unprofiled_reference_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Remove SDK activation, exports and loader injection before worker exec."""
    return {
        key: value
        for key, value in environment.items()
        if key not in PROFILER_LOADER_NAMES
        and not key.startswith(PROFILER_ENV_PREFIXES)
    }


def exec_scope_candidate(
    staging: Path, environment: dict[str, str]
) -> NoReturn:
    """Launch rocprofv3 after the reference worker starts, retaining private FDs.

    The controller writes this launcher before any candidate import. rocprofv3
    uses execvpe for one-pass collection, preserving the inheritable IPC pipes.
    """
    launch = json.loads((staging / SCOPE_LAUNCH_FILE).read_text())
    environment.update(launch["environment"])
    command = launch["command"]
    os.execvpe(command[0], command, environment)
