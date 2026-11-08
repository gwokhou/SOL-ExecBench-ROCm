"""Independent research coverage artifacts; no official scoring authority."""

from enum import StrEnum


class CoverageArtifactSchema(StrEnum):
    """Versioned coverage envelopes owned by the outer SOLAR bridge."""

    MATRIX = "sol_execbench.target_coverage.v1"
    OBSERVATION = "sol_execbench.solar_stage_observation.v1"
