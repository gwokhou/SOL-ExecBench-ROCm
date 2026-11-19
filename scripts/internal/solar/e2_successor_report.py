"""Reconcile E0 correctness attempts and keep independent and gate labels separate."""

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import Activity, Session
from scripts.internal.solar.e2_evidence_budget import admit_evidence_writer
from scripts.internal.solar.e2_successor_cost_report import build_cost_report
from scripts.internal.solar.e2_successor_labels import (
    NumericAssessment,
    NumericLabel,
    assess_numeric_call,
    tolerances,
)
from scripts.internal.solar.e2_successor_ledger import (
    AttemptState,
    Phase,
    SuccessorLedger,
    planned_sessions,
)
from sol_execbench.core.bench.judge.contracts import JudgeFamily, JudgeStrategy
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    canonical_json_bytes,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file

_MANDATORY = frozenset({JudgeFamily.G2, JudgeFamily.R0, JudgeFamily.L3})
_PRECISION = frozenset({JudgeFamily.G2, JudgeFamily.R2})
_CONTROLS = frozenset({JudgeStrategy.REFERENCE, JudgeStrategy.ALTERNATIVE})


class CorrectnessOutcome(StrEnum):
    """Outcomes supported by certified, actually reached calls."""

    DETECTED = "required_violation_detected"
    MISSED = "required_violation_missed"
    LEGAL_PASSED = "certified_control_passed"
    FALSE_REJECTED = "certified_control_rejected"
    CONTROL_OUTSIDE_TOLERANCE = "control_outside_tolerance"
    STRONGER_GUARANTEE_ONLY = (
        "stronger_guarantee_failed_under_original_tolerance"
    )
    NORM_STABILITY_WITNESS = "unstable_norm_witness"
    INDETERMINATE = "indeterminate"
    NOT_NUMERICALLY_REACHED = "numeric_stage_not_reached"
    DESCRIPTIVE = "descriptive"
    UNAVAILABLE = "attempt_evidence_unavailable"


def _mandatory_outcome(
    calls: Sequence[NumericAssessment],
) -> CorrectnessOutcome:
    violations = [c for c in calls if c.label is NumericLabel.VIOLATION]
    if any(c.original_numeric_passed is True for c in violations):
        return CorrectnessOutcome.MISSED
    if any(c.original_numeric_passed is False for c in violations):
        return CorrectnessOutcome.DETECTED
    return CorrectnessOutcome.INDETERMINATE


def _control_outcome(calls: Sequence[NumericAssessment]) -> CorrectnessOutcome:
    if all(
        c.label is NumericLabel.WITHIN_TOLERANCE
        and c.reference_disagreement is False
        for c in calls
    ):
        if any(c.original_numeric_passed is False for c in calls):
            return CorrectnessOutcome.FALSE_REJECTED
        if all(c.original_numeric_passed is True for c in calls):
            return CorrectnessOutcome.LEGAL_PASSED
    if any(c.label is NumericLabel.VIOLATION for c in calls):
        return CorrectnessOutcome.CONTROL_OUTSIDE_TOLERANCE
    return CorrectnessOutcome.INDETERMINATE


def classify_correctness(
    session: Session, calls: Sequence[NumericAssessment]
) -> CorrectnessOutcome:
    """Apply E0 contrast definitions to the independent and original observations."""
    if not calls:
        return CorrectnessOutcome.NOT_NUMERICALLY_REACHED
    if (
        session.strategy is JudgeStrategy.SEMANTIC
        and session.case.family in _MANDATORY
    ):
        return _mandatory_outcome(calls)
    if session.strategy in _CONTROLS:
        return _control_outcome(calls)
    if session.strategy is JudgeStrategy.NUMERICAL:
        if session.case.family in _PRECISION:
            if any(
                c.label is NumericLabel.WITHIN_TOLERANCE
                and c.stronger_guarantee_passed is False
                and c.original_numeric_passed is True
                for c in calls
            ):
                return CorrectnessOutcome.STRONGER_GUARANTEE_ONLY
            return CorrectnessOutcome.INDETERMINATE
        if session.case.family is JudgeFamily.L0:
            if any(
                c.label is NumericLabel.VIOLATION
                and c.stronger_guarantee_passed is False
                for c in calls
            ):
                return CorrectnessOutcome.NORM_STABILITY_WITNESS
            return CorrectnessOutcome.INDETERMINATE
    return CorrectnessOutcome.DESCRIPTIVE


def review_correctness_session(path: Path, session: Session) -> dict[str, Any]:
    """Recompute only labels from a captured session without executing a candidate."""
    receipt = load_json_value(path)
    if any(receipt[key] != value for key, value in session.metadata().items()):
        raise ValueError("E2 correctness receipt coordinates changed")
    capture = receipt.get("observations", {})
    assessments: list[NumericAssessment] = []
    details = []
    atol, rtol = tolerances(receipt["original_checks"])
    for index, row in enumerate(capture.get("observations", [])):
        artifact = f"call-{index:03d}.npz"
        if row["artifact"] != artifact:
            raise ValueError("E2 correctness capture order changed")
        source = path.parent / "judge-observations" / artifact
        assessment = assess_numeric_call(source, row, atol=atol, rtol=rtol)
        if row["operation"] == "correctness":
            assessments.append(assessment)
        details.append(
            {
                "artifact": artifact,
                "artifact_sha256": row["artifact_sha256"],
                "input_sha256": row["input_sha256"],
                "operation": row["operation"],
                "round_index": row["round_index"],
                **asdict(assessment),
            }
        )
    verdict = receipt["original_evaluator_verdict"]
    if verdict == "PASSED" and any(
        call.original_numeric_passed is False for call in assessments
    ):
        raise ValueError(
            "accepted E2 trace conflicts with its numeric call receipts"
        )
    return {
        "session_id": session.identity,
        "verdict": verdict,
        "outcome": classify_correctness(session, assessments),
        "calls": details,
        "receipt_sha256": sha256_file(path),
    }


def _expected_contrast(session: Session) -> str:
    if session.strategy in _CONTROLS:
        return "control"
    family = session.case.family
    if session.strategy is JudgeStrategy.SEMANTIC and family in _MANDATORY:
        return "mandatory_violation"
    if session.strategy is JudgeStrategy.NUMERICAL and family in _PRECISION:
        return "precision_witness"
    if session.strategy is JudgeStrategy.NUMERICAL and family is JudgeFamily.L0:
        return "norm_stability_witness"
    return "descriptive"


def _one_unit(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    first = repeats[0]
    complete = all(r["state"] == AttemptState.COMPLETED for r in repeats)
    outcomes = {r["outcome"] for r in repeats}
    verdicts = {r["verdict"] for r in repeats}
    return {
        "case_id": first["case_id"],
        "strategy": first["strategy"],
        "contrast": first["contrast"],
        "complete": complete,
        "repeat_consistent": len(outcomes) == len(verdicts) == 1
        if complete
        else None,
        "outcome": next(iter(outcomes))
        if complete and len(outcomes) == 1
        else None,
        "repeats": repeats,
    }


def _observed_criteria(
    contrasts: Mapping[str, Counter[str]], inconsistent_units: int
) -> dict[str, Any]:
    """Report contradicted criteria and coverage of the independently tested subset."""
    mandatory = contrasts["mandatory_violation"]
    control = contrasts["control"]
    return {
        "mandatory_certified_sessions": (
            mandatory[CorrectnessOutcome.DETECTED]
            + mandatory[CorrectnessOutcome.MISSED]
        ),
        "mandatory_missed_sessions": mandatory[CorrectnessOutcome.MISSED],
        "certified_control_sessions": (
            control[CorrectnessOutcome.LEGAL_PASSED]
            + control[CorrectnessOutcome.FALSE_REJECTED]
        ),
        "false_rejection_sessions": control[CorrectnessOutcome.FALSE_REJECTED],
        "precision_witness_sessions": contrasts["precision_witness"][
            CorrectnessOutcome.STRONGER_GUARANTEE_ONLY
        ],
        "norm_stability_witness_sessions": contrasts["norm_stability_witness"][
            CorrectnessOutcome.NORM_STABILITY_WITNESS
        ],
        "inconsistent_units": inconsistent_units,
    }


def summarize_correctness(
    reviewed: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Preserve each of 1,296 sessions and all 432 logical units in the report."""
    canonical = [
        item
        for item in planned_sessions(Phase.ONLINE)
        if item.session.activity is Activity.CORRECTNESS
    ]
    known = {item.session.identity for item in canonical}
    if not reviewed.keys() <= known:
        raise ValueError(
            "E2 correctness report received sessions outside its denominator"
        )
    groups: dict[tuple[object, object], list[dict[str, Any]]] = defaultdict(
        list
    )
    for item in canonical:
        metadata = item.metadata()
        observed = reviewed.get(item.session.identity, {})
        row = {
            "state": AttemptState.NOT_REACHED,
            "verdict": None,
            "outcome": CorrectnessOutcome.UNAVAILABLE,
            **observed,
            **metadata,
            "contrast": _expected_contrast(item.session),
        }
        groups[(metadata["case_id"], metadata["strategy"])].append(row)
    units = [_one_unit(repeats) for repeats in groups.values()]
    all_rows = [row for group in groups.values() for row in group]
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in all_rows:
        counters[row["contrast"]][row["outcome"]] += 1
    inconsistent_units = sum(u["repeat_consistent"] is False for u in units)
    return {
        "performance_authority": False,
        "logical_units": len(units),
        "planned_sessions": len(all_rows),
        "attempt_states": dict(Counter(r["state"] for r in all_rows)),
        "session_outcomes_by_contrast": {
            c: dict(n) for c, n in counters.items()
        },
        "logical_unit_outcomes": dict(
            Counter(u["outcome"] or "unresolved" for u in units)
        ),
        "inconsistent_units": inconsistent_units,
        "complete_units": sum(u["complete"] for u in units),
        "observed_judge_criteria": _observed_criteria(
            counters, inconsistent_units
        ),
        "units": units,
    }


def build_correctness_report(ledger: SuccessorLedger) -> dict[str, Any]:
    """Reconcile the completed phase receipts before inspecting any raw data."""
    online = ledger.reconcile(Phase.ONLINE)
    reviewed = {}
    for item, row in zip(planned_sessions(Phase.ONLINE), online, strict=True):
        if item.session.activity is not Activity.CORRECTNESS:
            continue
        observed = None
        if row.get("evidence_path"):
            path = (
                ledger.root
                / Phase.ONLINE
                / item.directory
                / row["evidence_path"]
            )
            observed = review_correctness_session(path, item.session)
            if observed["verdict"] != row["verdict"]:
                raise ValueError(
                    "E2 original verdict disagrees with committed receipt"
                )
        reviewed[item.session.identity] = {
            **(observed or {}),
            "state": row["state"],
            "verdict": row["verdict"],
            "stop_reason": row.get("stop_reason"),
        }
    return summarize_correctness(reviewed)


def write_reports(ledger: SuccessorLedger) -> dict[str, Any]:
    """Write reproducible correctness and cost evidence for the existing campaign."""
    correctness = build_correctness_report(ledger)
    cost = build_cost_report(ledger)
    destination = ledger.root / "report"
    if destination.exists():
        raise FileExistsError(destination)
    payload = {"correctness.json": correctness, "cost.json": cost}
    planned = (
        sum(len(canonical_json_bytes(report)) for report in payload.values())
        + 1024**2
    )
    admit_evidence_writer(ledger.root, writer_limit_bytes=planned)
    destination.mkdir(exist_ok=False)
    for filename, report in payload.items():
        atomic_write_json_value(destination / filename, report)
    result = {
        "roster_sha256": ledger.roster_sha256,
        "output_sha256": {
            filename: sha256_file(destination / filename)
            for filename in payload
        },
        "complete_correctness_units": correctness["complete_units"],
        "comparable_full_cost_pairs": cost["comparable_pairs"],
        "performance_authority": False,
    }
    atomic_write_json_value(destination / "report-index.json", result)
    return result


def main() -> None:
    """Reconcile completed and interrupted receipts without running workloads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    opts = parser.parse_args()
    ledger = SuccessorLedger(
        opts.evidence_root, freeze_sha256=sha256_file(opts.freeze)
    )
    write_reports(ledger)


if __name__ == "__main__":
    main()
