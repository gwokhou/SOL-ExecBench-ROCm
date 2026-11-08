"""Content-bound raw arrays owned by a development-report envelope."""

import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from sol_execbench.core.bench.judge.inputs import JudgeInputs
from sol_execbench.core.integrity import (
    sha256_bytes,
    stable_json_checksum,
)
from sol_execbench.core.integrity.source_tree import source_tree_manifest


def tensor_identity(values: tuple[torch.Tensor, ...]) -> str:
    """Hash quantized CPU bytes, shapes, and dtypes in input order."""
    return stable_json_checksum(
        [
            {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "bytes": sha256_bytes(
                    value.detach()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                ),
            }
            for value in values
        ]
    )


def source_identity() -> dict[str, str]:
    """Bind every local diagnostic source, including candidate interventions."""
    return source_tree_manifest(Path(__file__).resolve().parents[4])


def input_scale(values: tuple[torch.Tensor, ...]) -> torch.Tensor | None:
    """Return the preregistered absolute-contribution scale, without a new threshold."""
    if len(values) == 2:
        return values[0].double().abs() @ values[1].double().abs()
    if len(values) == 1:
        return values[0].double().abs().sum(-1, keepdim=True)
    return None


def save_case_arrays(
    output: Path,
    inputs: JudgeInputs,
    reference: torch.Tensor,
    expected: torch.Tensor | None,
    exact: tuple[Fraction, ...] | tuple[tuple[str, str], ...],
    candidates: dict[str, torch.Tensor],
) -> tuple[Path, ...]:
    """Atomically save non-pickled arrays and exact rational/interval strings."""
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        f"input_{index}": value.detach().double().numpy()
        for index, value in enumerate(inputs.values)
    }
    for name, values in inputs.construction.items():
        arrays[f"construction_{name}"] = np.array(values, dtype=np.int64)
    arrays["input_dtypes"] = np.array(
        [str(value.dtype) for value in inputs.values]
    )
    arrays["witness_columns"] = np.array(inputs.witness_columns, dtype=np.int64)
    arrays["reference"] = reference.detach().double().numpy()
    scale = input_scale(inputs.values)
    if scale is not None:
        arrays["sum_absolute_contributions"] = scale.numpy()
    elif len(inputs.values) == 4:
        x = inputs.values[0].double()
        arrays["actual_mean"] = x.mean(-1).numpy()
        arrays["actual_population_variance"] = (
            ((x - x.mean(-1, keepdim=True)) ** 2).mean(-1).numpy()
        )
    if expected is not None:
        arrays["oracle_rounded"] = expected.double().numpy()
    arrays["oracle_exact"] = np.array([str(item) for item in exact])
    for strategy, value in candidates.items():
        arrays[f"candidate_{strategy}"] = value.detach().double().numpy()
    target = output / "arrays.npz"
    with tempfile.NamedTemporaryFile(
        dir=output, suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return (target,)
