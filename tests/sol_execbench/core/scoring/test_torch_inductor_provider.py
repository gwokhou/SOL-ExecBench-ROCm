from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from sol_execbench.core.data.workload import ToleranceSpec, Workload


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "internal" / "run_torch_inductor_provider.py"
spec = importlib.util.spec_from_file_location("torch_inductor_provider", SCRIPT_PATH)
assert spec is not None
provider = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = provider
spec.loader.exec_module(provider)


def _workload(required_matched_ratio: float) -> Workload:
    return Workload(
        axes={"n": 100},
        inputs={},
        uuid="provider-tolerance",
        tolerance=ToleranceSpec(
            max_atol=0.0,
            max_rtol=0.05,
            required_matched_ratio=required_matched_ratio,
        ),
    )


def test_provider_uses_workload_matched_ratio_for_compiled_output() -> None:
    eager = torch.ones(100)
    actual = eager.clone()
    actual[:9] = 1.2

    provider._assert_correct_output(actual, eager, _workload(0.9))


def test_provider_rejects_compiled_output_below_workload_matched_ratio() -> None:
    eager = torch.ones(100)
    actual = eager.clone()
    actual[:11] = 1.2

    with pytest.raises(AssertionError, match="exceeds workload tolerance"):
        provider._assert_correct_output(actual, eager, _workload(0.9))
