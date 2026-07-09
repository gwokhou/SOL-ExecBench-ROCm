# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sol_execbench.core.platform.compatibility import RocmCompatibilityMatrixReport


def load_matrix_report(
    path_or_payload: str | Path | dict[str, Any],
) -> RocmCompatibilityMatrixReport:
    """Load and validate a ROCm Compatibility Matrix report."""

    if isinstance(path_or_payload, str | Path):
        payload = json.loads(Path(path_or_payload).read_text(encoding="utf-8"))
    else:
        payload = path_or_payload
    try:
        return RocmCompatibilityMatrixReport.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid Matrix report: {exc}") from exc
