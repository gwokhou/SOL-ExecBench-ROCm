"""Offline diagnosis of captured incumbent-mode outputs; never execute candidates."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.internal.solar.e2_contracts import Session
from scripts.internal.solar.e2_observation import (
    Observation,
    diagnose,
    encode_observation,
)
from sol_execbench.core.bench.judge.contracts import (
    JudgeInstance,
    load_case_problem,
)
from sol_execbench.core.bench.judge.inputs import JudgeInputs
from sol_execbench.core.bench.judge.interval_oracle import OracleUncertainError
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_bytes, sha256_file


def decode_observation(path: Path, coordinate: dict[str, Any]) -> Observation:
    """Recover actual quantized tensors from non-pickled, bounded session arrays."""
    with np.load(path, allow_pickle=False) as arrays:
        dtypes = {
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
        }
        values = tuple(
            torch.tensor(arrays[f"input_{index}"], dtype=dtypes[str(dtype)])
            for index, dtype in enumerate(arrays["input_dtypes"])
        )
        inputs = JudgeInputs(
            values=values,
            witness_columns=tuple(int(i) for i in arrays["witness_columns"]),
            construction={
                key.removeprefix("construction_"): tuple(
                    int(i) for i in arrays[key]
                )
                for key in arrays.files
                if key.startswith("construction_")
            },
        )
        dtype = dtypes[str(arrays["output_dtype"])]
        return Observation(
            inputs=inputs,
            reference=torch.tensor(arrays["reference"], dtype=dtype),
            actual=torch.tensor(arrays["actual"], dtype=dtype),
            coordinate=coordinate,
        )


def diagnose_incumbent(
    repository: Path, output: Path, session: Session
) -> dict[str, Any]:
    """Keep oracle costs outside the baseline session wall and preserve raw arrays."""
    source = output / "judge-observations"
    payload = json.loads((source / "observations.json").read_text())
    definition, workload = load_case_problem(
        repository, replace(session.case, instance=JudgeInstance.DEVELOPMENT_0)
    )
    results = []
    stop_reason = None
    for row in payload["observations"]:
        path = source / row["artifact"]
        if sha256_file(path) != row["artifact_sha256"]:
            raise ValueError("incumbent observation artifact identity changed")
        observation = decode_observation(path, row)
        try:
            diagnostic, arrays = diagnose(
                session.case,
                session.strategy,
                observation,
                definition,
                workload,
            )
        except (OracleUncertainError, TimeoutError, ValueError) as exc:
            stop_reason = f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    **row,
                    "diagnostic_status": "oracle_uncertain",
                    "message": stop_reason,
                }
            )
            break
        _, encoded = encode_observation(observation, arrays)
        target = path.with_suffix(".diagnosed.npz")
        target.write_bytes(encoded)
        results.append(
            {
                **row,
                **diagnostic,
                "diagnostic_artifact": target.name,
                "diagnostic_artifact_sha256": sha256_bytes(encoded),
            }
        )
    result = {
        "observations": results,
        "stop_reason": stop_reason,
        "cost_authority": "offline_not_charged_to_baseline_session",
    }
    atomic_write_json_value(output / "offline-diagnostic.json", result)
    return result
