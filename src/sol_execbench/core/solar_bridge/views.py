# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Read view summaries from the exact artifact accepted by the outer worker."""

from pathlib import Path

from sol_execbench.core.solar_bridge.models import SolarAnalysisOutcome
from solar.analysis.views import read_analysis_views
from solar.artifacts import MAX_ARTIFACT_BYTES, sha256_bytes
from solar.artifacts.bundle import read_regular_file
from solar.artifacts.document import parse_yaml_artifact
from solar.artifacts.names import ANALYSIS_FILENAME
from solar.schema_versions import SOLAR_ANALYSIS_SCHEMA_VERSION


def load_outcome_views(outcome: SolarAnalysisOutcome) -> dict[str, object]:
    """Validate the accepted analysis digest and expose every view and reason."""
    matches = [
        item for item in outcome.artifacts if item["path"] == ANALYSIS_FILENAME
    ]
    if outcome.output_dir is None or len(matches) != 1:
        raise ValueError("SOLAR outcome lacks one bound analysis artifact")
    path = Path(outcome.output_dir) / ANALYSIS_FILENAME
    payload = read_regular_file(path, max_bytes=MAX_ARTIFACT_BYTES)
    if sha256_bytes(payload) != matches[0]["sha256"]:
        raise ValueError("SOLAR analysis changed after worker acceptance")
    analysis = parse_yaml_artifact(path, payload)
    if analysis.get("schema_version") != SOLAR_ANALYSIS_SCHEMA_VERSION:
        raise ValueError("SOLAR analysis schema is not current")
    views = read_analysis_views(
        analysis.require_mapping("metadata").get("views")
    )
    return {
        kind.value: view.model_dump(mode="json") for kind, view in views.items()
    }
