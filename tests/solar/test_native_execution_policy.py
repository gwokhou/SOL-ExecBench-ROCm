"""Standalone SOLAR GPU callers must supply their existing lifecycle context."""

from contextlib import nullcontext
from functools import partial

import pytest

from solar.contracts import (
    AnalysisExecutionPolicy,
    AnalysisRequest,
    ConversionRequest,
)
from solar.verification.contracts import VerificationPolicy


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "hip:0"])
def test_gpu_request_requires_context_not_only_private_lock(tmp_path, device):
    conversion = ConversionRequest(
        analysis_id="cpu-policy-fixture",
        reference=lambda x: x,
        input_factory=lambda _: (),
        reference_name="fixture",
        reference_sha256="a" * 64,
        verification=VerificationPolicy(device=device, atol=1e-5, rtol=1e-5),
    )
    make_request = partial(
        AnalysisRequest,
        conversion=conversion,
        architecture="RX_9060_XT",
        output_dir=tmp_path,
    )
    if device == "cpu":
        make_request()
    else:
        with pytest.raises(ValueError, match="native device_stage_context"):
            make_request(
                execution_policy=AnalysisExecutionPolicy(
                    device_stage_lock_path=tmp_path / "private.lock"
                ),
            )
        make_request(
            execution_policy=AnalysisExecutionPolicy(
                device_stage_context=nullcontext
            ),
        )
