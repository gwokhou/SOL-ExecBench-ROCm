# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Isolated ShapesSpec experiment; does not register a production SOLAR IR path."""

import argparse
import time
import tracemalloc
from enum import StrEnum
from pathlib import Path

import torch
from torch.fx.experimental.dynamic_spec import ShapesSpec, ShapeVar, TensorSpec
from torch.fx.experimental.proxy_tensor import make_fx

from sol_execbench.core.data.json_utils import atomic_write_json_value

MIN_BATCH = 2
MAX_BATCH = 16
TRACE_BATCH = 8
WIDTH = 3
DIVISOR = 4
REPLAY_BATCHES = (2, 8, 16, 7, 1, 17)
DIVISIBLE_REPLAY_BATCHES = (4, 8, 16, 12, 2, 7, 17)


class ShapeCase(StrEnum):
    """Fixed development cases, independent of the held-out benchmark corpus."""

    SHARED = "shared_batch"
    DERIVED = "derived_dimension"
    RANGE = "bounded_range"
    DIVISIBLE = "divisibility"
    BRANCH = "shape_branch"
    NONZERO = "nonzero"


def operation(
    case: ShapeCase, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Evaluate an intentionally small shape-sensitive expression."""
    match case:
        case ShapeCase.SHARED:
            return x + y
        case ShapeCase.DERIVED:
            return torch.cat((x, x)) + y
        case ShapeCase.DIVISIBLE:
            return x.reshape(x.shape[0] // DIVISOR, DIVISOR, WIDTH).sum(1)
        case ShapeCase.BRANCH:
            return x.sin() if x.shape[0] > TRACE_BATCH else x.cos()
        case ShapeCase.NONZERO:
            return x.nonzero()
        case ShapeCase.RANGE:
            return x.sum(1)


def arguments(case: ShapeCase, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic concrete values including data-dependent zeros."""
    x = torch.arange(batch * WIDTH, dtype=torch.float32).reshape(batch, WIDTH)
    rows = batch * 2 if case is ShapeCase.DERIVED else batch
    y = torch.ones(rows, WIDTH)
    return x, y


def constraint(case: ShapeCase) -> dict[str, object]:
    """Own an inspectable constraint description, never serialize SymBool repr."""
    return {
        "batch": {"min": MIN_BATCH, "max": MAX_BATCH},
        "x": ["batch", WIDTH],
        "y": [{"multiply": ["batch", 2]}, WIDTH]
        if case is ShapeCase.DERIVED
        else ["batch", WIDTH],
        "divisible_by": DIVISOR if case is ShapeCase.DIVISIBLE else None,
        "authority": "finite_concrete_replay_only",
    }


def specification(case: ShapeCase) -> ShapesSpec:
    """Translate the fixed owned constraints to the experimental upstream API."""
    batch = ShapeVar("batch", min=MIN_BATCH, max=MAX_BATCH)
    y_batch = batch * 2 if case is ShapeCase.DERIVED else batch
    return ShapesSpec(
        {"x": TensorSpec([batch, WIDTH]), "y": TensorSpec([y_batch, WIDTH])},
        assumptions=[batch % DIVISOR == 0]
        if case is ShapeCase.DIVISIBLE
        else [],
    )


def replay(
    graph: torch.fx.GraphModule, case: ShapeCase
) -> list[dict[str, object]]:
    """Record lawful agreement and raw graph behavior on constraint violations."""
    results = []
    batches = (
        DIVISIBLE_REPLAY_BATCHES
        if case is ShapeCase.DIVISIBLE
        else REPLAY_BATCHES
    )
    inputs = [(batch, True) for batch in batches]
    if case in {ShapeCase.SHARED, ShapeCase.DERIVED}:
        inputs.append((TRACE_BATCH, False))
    for batch, matching_dimension in inputs:
        lawful = (
            matching_dimension
            and MIN_BATCH <= batch <= MAX_BATCH
            and (case is not ShapeCase.DIVISIBLE or batch % DIVISOR == 0)
        )
        row: dict[str, object] = {
            "batch": batch,
            "matching_dimension": matching_dimension,
            "lawful": lawful,
        }
        args = arguments(case, batch)
        if not matching_dimension:
            args = (args[0], torch.ones(args[1].shape[0] + 1, WIDTH))
        try:
            actual = graph(*args)
            torch.testing.assert_close(actual, operation(case, *args))
            row["result"] = "matches_reference"
        except (RuntimeError, ValueError, AssertionError, TypeError) as exc:
            row["result"] = "rejected_or_mismatch"
            row["error_type"] = type(exc).__name__
        results.append(row)
    return results


def trace_case(case: ShapeCase, mode: str) -> dict[str, object]:
    """Measure tracing and replay without claiming symbolic proof or GPU speedup."""

    def function(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return operation(case, x, y)

    result: dict[str, object] = {
        "case": case,
        "mode": mode,
        "constraints": constraint(case),
    }
    tracemalloc.start()
    started = time.perf_counter()
    try:
        graph = make_fx(
            function,
            tracing_mode=mode,
            dynamic_shapes=specification(case) if mode == "fake" else None,
        )(*arguments(case, TRACE_BATCH))
        result["trace_seconds"] = time.perf_counter() - started
        result.update(
            status="traced", graph=graph.code, replays=replay(graph, case)
        )
    except (RuntimeError, ValueError, AssertionError, TypeError) as exc:
        result.update(
            status="trace_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:2000],
        )
    finally:
        result["trace_and_replay_seconds"] = time.perf_counter() - started
        result["python_heap_peak_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return result


def main() -> None:
    """Write a bounded experiment receipt outside production evidence schemas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json_value(
        args.output,
        {
            "experiment": "shapes_spec_development",
            "torch": torch.__version__,
            "production_ir_changed": False,
            "results": [
                trace_case(case, mode)
                for case in ShapeCase
                for mode in ("real", "fake")
            ],
        },
    )


if __name__ == "__main__":
    main()
