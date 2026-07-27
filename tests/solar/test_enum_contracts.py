from __future__ import annotations

import json

from solar.contracts import SolarAnalysisStatus, SolarStage


def test_solar_strenum_members_serialize_without_value_extraction() -> None:
    payload = {
        "analysis": SolarAnalysisStatus.ANALYZED,
        "stage": SolarStage.FORMAL_ANALYSIS,
    }

    assert json.loads(json.dumps(payload)) == {
        "analysis": "analyzed",
        "stage": "formal_analysis",
    }
