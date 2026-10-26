"""Validate the bounded worker receipt once, before baseline aggregation."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
    model_validator,
)

from scripts.internal.solar.base01_samples import Family, Role, Sample
from solar.contracts import SolarReadinessStatus
from solar.ir.contracts import IRPath

OBSERVATION_LIMIT_BYTES = 1_048_576


class AccountingStatus(StrEnum):
    """Observed accounting outcomes, independent of conversion readiness."""

    NOT_RUN = "not_run"
    FAILED = "failed"
    COMPLETED = "completed"


class ReceiptFailure(StrEnum):
    """Worker protocol failures retained separately from SOLAR refusals."""

    MISSING = "missing_observation"
    INVALID = "invalid_observation"
    TOO_LARGE = "observation_too_large"


class ObservationDataError(ValueError):
    """An invalid child receipt, without including its potentially large body."""

    def __init__(self, reason: ReceiptFailure) -> None:
        """Store a bounded reason for the failed receipt boundary."""
        self.reason = reason
        super().__init__(reason.value)


class SampleIdentity(BaseModel):
    """Only identity fields are consumed here; full inputs remain in evidence."""

    model_config = ConfigDict(extra="allow")
    family: Family
    role: Role


class ReadinessReceipt(BaseModel):
    """The production readiness fields used by the baseline denominator."""

    model_config = ConfigDict(extra="allow")
    status: SolarReadinessStatus
    reason_code: StrictStr | None


class MathComparison(BaseModel):
    """A numeric observation never converts missing work into a matched zero."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    expected: int | float
    observed: int | float | None
    matches: StrictBool

    @model_validator(mode="after")
    def validate_match(self) -> MathComparison:
        """Reject contradictory results before they reach aggregation."""
        if self.matches != (self.expected == self.observed):
            raise ValueError("math comparison flag contradicts its operands")
        return self


class Observation(BaseModel):
    """Validated aggregation fields; untouched diagnostic fields are retained."""

    model_config = ConfigDict(extra="allow")
    sample: SampleIdentity
    ir_path: IRPath
    readiness: ReadinessReceipt
    accounting_status: AccountingStatus
    publication_eligible: Literal[False]
    real_mapper: Literal["not_run"]
    total: dict[str, object] | None = None
    math_comparison: dict[str, MathComparison] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_states(self) -> Observation:
        """Require evidence consistent with conversion and accounting states."""
        if self.readiness.status is not SolarReadinessStatus.READY:
            if not self.readiness.reason_code:
                raise ValueError("failed conversion has no reason")
            if self.accounting_status is not AccountingStatus.NOT_RUN:
                raise ValueError("unverified conversion was counted")
        if (
            self.accounting_status is AccountingStatus.COMPLETED
            and self.total is None
        ):
            raise ValueError("completed accounting has no totals")
        return self


def load_observation(
    path: Path, sample: Sample, ir_path: IRPath
) -> Observation:
    """Limit input bytes before JSON parsing and bind the returned slot identity."""
    with path.open("rb") as stream:
        payload = stream.read(OBSERVATION_LIMIT_BYTES + 1)
    if len(payload) > OBSERVATION_LIMIT_BYTES:
        raise ObservationDataError(ReceiptFailure.TOO_LARGE)
    try:
        result = Observation.model_validate_json(payload)
    except ValidationError as exc:
        raise ObservationDataError(ReceiptFailure.INVALID) from exc
    if (
        result.sample.family is not sample.family
        or result.sample.role is not sample.role
        or result.ir_path is not ir_path
    ):
        raise ObservationDataError(ReceiptFailure.INVALID)
    return result
