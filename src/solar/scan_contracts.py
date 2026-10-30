# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public scan orchestration request over the profile-independent grid contract."""

from pydantic import model_validator

from solar.analysis.demand_contracts import ComputationScenario
from solar.analysis.scan_contracts import ScanDefinition, ScanReference
from solar.reuse_contracts import ReuseJob


class ScanRequest(ScanDefinition):
    """A pinned program, a fixed architecture grid and an optional prior attempt."""

    job: ReuseJob
    resume: ScanReference | None = None

    @model_validator(mode="after")
    def _hypothetical_only(self) -> ScanRequest:
        if self.job.projection.require_verified_audit:
            raise ValueError(
                "hypothetical scans cannot require formal profile audit"
            )
        if (
            self.computation is ComputationScenario.RESTRICTED
            and self.job.projection.restricted is None
        ):
            raise ValueError(
                "restricted scan requires explicit computation constraints"
            )
        return self
