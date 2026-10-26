"""Current read-proof fixtures for artifact reader tests, not numerical oracles."""

from dataclasses import asdict
from typing import Any

from solar.analysis.access_regions import DEFAULT_ACCESS_BUDGET
from solar.analysis.views import VIEW_ASSUMPTIONS, AnalysisView
from solar.ir.access import AccessReason
from solar.precision import BYTES_PER_ELEMENT


def exact_reads(elements: int = 0, dtype: str = "fp32") -> dict[str, Any]:
    """Represent a distinct fp32 base; empty graphs have an exact empty union."""
    bases = (
        {}
        if not elements
        else {
            "input": {
                "shape": [elements],
                "dtype": dtype,
                "elements": elements,
                "bytes": elements * BYTES_PER_ELEMENT[dtype],
                "quality": "exact",
                "reason": "static_logical_map",
            }
        }
    )
    return {
        "quality": "exact",
        "reasons": ["static_logical_map"] if bases else [],
        "scope": "external_reads_fixed_operator_graph",
        "elements": elements,
        "bytes": elements * BYTES_PER_ELEMENT[dtype],
        "bases": bases,
        "budget": asdict(DEFAULT_ACCESS_BUDGET),
    }


def unavailable_reads() -> dict[str, Any]:
    """Describe synthetic graphs that have no external-access proof."""
    return {
        **exact_reads(),
        "quality": "unavailable",
        "reasons": [AccessReason.UNKNOWN_READS.value],
        "elements": None,
        "bytes": None,
    }


def unavailable_views() -> dict[str, Any]:
    """Keep reader fixtures explicit about absent model and capacity evidence."""
    return {
        kind.value: {
            "status": "unavailable",
            "reason": "external_reads_unproved",
            "seconds": None,
            "io_bytes": None,
            "resource_seconds": {},
            "operator_seconds": {},
            "read_quality": "unavailable",
            "assumptions": list(VIEW_ASSUMPTIONS[kind]),
        }
        for kind in AnalysisView
    }
