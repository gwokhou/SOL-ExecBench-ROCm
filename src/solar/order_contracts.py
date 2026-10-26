# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public contraction-search request over an exactly pinned verified source."""

from pydantic import model_validator

from solar.analysis.demand_contracts import ComputationScenario
from solar.analysis.order_contracts import (
    OrderBudget,
    OrderContract,
    OrderReference,
)
from solar.analysis.views import AnalysisView
from solar.errors import SolarError
from solar.reuse_contracts import ReuseJob


class OrderPreparationTimeoutError(SolarError):
    """No finite plan was constructed before the preparation deadline."""

    reason_code = "contraction_search_preparation_timeout"


class OrderSearchRequest(OrderContract):
    """Search one path/profile/model, always retaining the original association."""

    job: ReuseJob
    view: AnalysisView
    computation: ComputationScenario = ComputationScenario.DENSE
    budget: OrderBudget = OrderBudget()
    resume: OrderReference | None = None

    @model_validator(mode="after")
    def _search_constraints(self) -> OrderSearchRequest:
        if self.job.source.verification.device != "cpu":
            raise ValueError(
                "contraction search currently supports CPU verification only"
            )
        if not self.job.source.verification.verify_gradients:
            raise ValueError(
                "contraction search requires the original gradient verification policy"
            )
        if self.job.projection.require_verified_audit:
            raise ValueError(
                "exploratory contraction search cannot request publication qualification"
            )
        if (
            self.computation is ComputationScenario.RESTRICTED
            and self.job.projection.restricted is None
        ):
            raise ValueError(
                "restricted search requires explicit computation constraints"
            )
        return self
