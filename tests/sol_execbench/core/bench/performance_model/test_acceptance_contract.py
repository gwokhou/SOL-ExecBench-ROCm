"""CPU-safe structural checks for the held-out acceptance denominator."""

from sol_execbench.core.bench.performance_model.acceptance import (
    DiagnosticAcceptanceManifest,
)


def test_acceptance_manifest_requires_exact_held_out_denominator() -> None:
    cases = DiagnosticAcceptanceManifest.model_json_schema()["properties"][
        "cases"
    ]

    assert cases["minItems"] == 220
    assert cases["maxItems"] == 220
