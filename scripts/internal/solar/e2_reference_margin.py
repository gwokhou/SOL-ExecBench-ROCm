"""Bound the numeric margin of one E2 call under observed reference error."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceMargin:
    """Bounds on the minimum all-element tolerance margin for this call."""

    lower: float
    upper: float
    maximum_reference_error: float


def reference_margin(
    actual: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
    oracle: npt.NDArray[np.float64],
    *,
    atol: float,
    rtol: float,
) -> ReferenceMargin:
    """Bound the margin over the empirical error envelope around the oracle.

    Each reference element can range within its measured distance from the
    certified rounded oracle. The margin can change by at most 1 + rtol times
    that distance. The envelope applies only to this retained invocation.
    A strictly positive lower bound certifies the positive sign and a strictly
    negative upper bound certifies the negative sign. A bound touching zero
    leaves the experiment's strict-margin condition unresolved.
    """
    if not (
        actual.size
        and actual.shape == reference.shape == oracle.shape
        and all(
            values.dtype == np.float64 and bool(np.isfinite(values).all())
            for values in (actual, reference, oracle)
        )
        and np.isfinite(atol)
        and np.isfinite(rtol)
        and atol >= 0
        and rtol >= 0
    ):
        raise ValueError(
            "E2 margin requires congruent finite FP64 arrays and tolerances"
        )

    # Outward rounding prevents a floating-point boundary from acquiring a
    # strict sign. Distances between equal values are exactly zero.
    reference_error = np.abs(reference - oracle)
    error_limit = np.where(
        reference_error == 0,
        0,
        np.nextafter(reference_error, np.inf),
    )
    slack = np.nextafter(np.nextafter(1 + rtol, np.inf) * error_limit, np.inf)
    deviation = np.abs(actual - oracle)
    relative_tolerance = rtol * np.abs(oracle)
    low_tolerance = np.nextafter(
        atol + np.nextafter(relative_tolerance, -np.inf), -np.inf
    )
    high_tolerance = np.nextafter(
        atol + np.nextafter(relative_tolerance, np.inf), np.inf
    )
    lower = np.nextafter(
        np.nextafter(low_tolerance - np.nextafter(deviation, np.inf), -np.inf)
        - slack,
        -np.inf,
    )
    upper = np.nextafter(
        np.nextafter(high_tolerance - np.nextafter(deviation, -np.inf), np.inf)
        + slack,
        np.inf,
    )
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("E2 margin arithmetic exceeded the finite range")
    return ReferenceMargin(
        lower=float(lower.min()),
        upper=float(upper.min()),
        maximum_reference_error=float(error_limit.max()),
    )
