"""Test lossless aliases for E2 arrays and older raw observation archives."""

import io

import numpy as np
import pytest

from scripts.internal.solar.e2_archive_arrays import (
    compact_output_arrays,
    read_output_arrays,
)


@pytest.mark.parametrize("compact", [False, True], ids=["legacy", "compact"])
def test_observed_outputs_are_reconstructed_byte_for_byte(compact):
    reference = np.array([13.0, -0.0])
    outputs = {
        "reference": reference,
        "actual": reference.copy(),
        "oracle_rounded": np.array([13.0, 0.0]),
    }
    stored = compact_output_arrays(outputs) if compact else outputs
    if compact:
        assert "actual" not in stored
        assert "oracle_rounded" in stored
    with io.BytesIO() as payload:
        np.savez_compressed(payload, allow_pickle=False, **stored)
        payload.seek(0)
        with np.load(payload, allow_pickle=False) as archive:
            resolved = read_output_arrays(archive, with_oracle=True)
    assert set(resolved) == set(outputs)
    for name in resolved:
        assert resolved[name].dtype == outputs[name].dtype
        assert resolved[name].shape == outputs[name].shape
        assert resolved[name].tobytes() == outputs[name].tobytes()


def test_invalid_output_aliases_are_rejected():
    with io.BytesIO() as payload:
        np.savez_compressed(
            payload,
            reference=np.array([1.0]),
            e2_output_aliases=np.array([["actual", "oracle_rounded"]]),
        )
        payload.seek(0)
        with (
            np.load(payload, allow_pickle=False) as archive,
            pytest.raises(ValueError, match="invalid output alias"),
        ):
            read_output_arrays(archive)
