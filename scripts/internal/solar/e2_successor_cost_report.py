"""Report all E0 cost pairs using only actual evaluator and replay receipts."""

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts.internal.solar.e2_contracts import Activity, Mode
from scripts.internal.solar.e2_successor_ledger import (
    AttemptState,
    Phase,
    SuccessorLedger,
    planned_sessions,
)
from sol_execbench.core.data.json_utils import load_json_value
from sol_execbench.core.integrity import sha256_file

_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)


def _components(receipt: dict[str, Any]) -> dict[str, Any]:
    capture = receipt.get("observations") or {}
    client = receipt.get("client_cost") or {}
    raw_calls = capture.get("observations")
    return {
        "candidate_calls": len(raw_calls) if raw_calls is not None else None,
        "reference_calls": capture.get("reference_calls"),
        "prepare_wall_seconds": receipt.get("prepare_wall_seconds"),
        "compile_seconds": receipt.get("compile_seconds"),
        "reference_ipc_seconds": client.get(
            "reference_ipc_wall_seconds_including_oracle"
        ),
        "reference_service_seconds": capture.get(
            "reference_prepare_service_wall_seconds"
        ),
        "checks_seconds": capture.get("current_checks_seconds"),
        "inline_oracle_wall_seconds": sum(
            r.get("oracle_wall_seconds", 0) for r in raw_calls
        )
        if raw_calls is not None
        else None,
        "inline_oracle_cpu_seconds": capture.get("oracle_cpu_seconds"),
        "capture_wall_seconds": capture.get("capture_wall_seconds"),
        "evidence_encoding_seconds": sum(
            r.get("evidence_encoding_wall_seconds", 0) for r in raw_calls
        )
        if raw_calls is not None
        else None,
        "session_wall_seconds": receipt.get("session_wall_seconds"),
        "candidate_rss_bytes": client.get("candidate_peak_rss_bytes"),
        "reference_rss_bytes": capture.get("reference_worker_peak_rss_bytes"),
        "scratch": receipt.get("scratch"),
    }


def _complete_call_path(row: dict[str, Any]) -> bool:
    metrics = row.get("components") or {}
    wall = metrics.get("session_wall_seconds")
    return bool(
        row["state"] == AttemptState.COMPLETED
        and row["verdict"] == "PASSED"
        and metrics.get("candidate_calls") == 190
        and metrics.get("reference_calls") == 371
        and isinstance(wall, (int, float))
        and math.isfinite(wall)
        and wall > 0
    )


def _pair(repeat: int, sides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    incumbent = sides[Mode.INCUMBENT]
    diagnostic = sides[Mode.DIAGNOSTIC]
    comparable = _complete_call_path(incumbent) and _complete_call_path(
        diagnostic
    )
    delta = ratio = None
    if comparable:
        plain = incumbent["components"]["session_wall_seconds"]
        observed = diagnostic["components"]["session_wall_seconds"]
        delta, ratio = observed - plain, observed / plain
    return {
        "repeat": repeat,
        "modes": sides,
        "both_full_call_paths": comparable,
        "online_wall_difference_seconds": delta,
        "online_wall_ratio": ratio,
    }


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    numbers = np.quantile(values, _QUANTILES, method="linear")
    return dict(zip(map(str, _QUANTILES), map(float, numbers), strict=True))


def summarize_costs(
    reviewed: Mapping[str, dict[str, Any]],
    replayed: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    """Preserve all cost blocks, modes and early or unavailable measurements."""
    grouped: dict[tuple[str, str], dict[int, dict[str, dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    plan = tuple(
        item
        for item in planned_sessions(Phase.ONLINE)
        if item.session.activity is Activity.COST
    )
    if not reviewed.keys() <= {
        item.session.identity for item in plan
    } or not replayed.keys() <= {
        item.session.identity
        for item in plan
        if item.session.mode is Mode.INCUMBENT
    }:
        raise ValueError(
            "E2 cost report received sessions outside its denominator"
        )
    for item in plan:
        session = item.session
        side = {
            "state": AttemptState.NOT_REACHED,
            "verdict": None,
            "components": None,
            **reviewed.get(session.identity, {}),
            **item.metadata(),
        }
        if session.mode is Mode.INCUMBENT:
            side["offline"] = {
                "state": AttemptState.NOT_REACHED,
                **replayed.get(session.identity, {}),
            }
        grouped[(session.case.logical_id, session.strategy.value)][
            session.repeat
        ][session.mode.value] = side
    blocks: list[dict[str, Any]] = []
    for (case_id, strategy), repeats in grouped.items():
        pairs = [_pair(repeat, modes) for repeat, modes in repeats.items()]
        compared = [pair for pair in pairs if pair["both_full_call_paths"]]
        blocks.append(
            {
                "case_id": case_id,
                "strategy": strategy,
                "pairs": pairs,
                "comparable_pairs": len(compared),
                "wall_difference_quantiles_seconds": _quantiles(
                    [p["online_wall_difference_seconds"] for p in compared]
                ),
                "wall_ratio_quantiles": _quantiles(
                    [p["online_wall_ratio"] for p in compared]
                ),
            }
        )
    return {
        "performance_authority": False,
        "planned_blocks": len(blocks),
        "planned_pairs": sum(len(b["pairs"]) for b in blocks),
        "comparable_pairs": sum(b["comparable_pairs"] for b in blocks),
        "offline_states": dict(
            Counter(
                pair["modes"][Mode.INCUMBENT]["offline"]["state"]
                for b in blocks
                for pair in b["pairs"]
            )
        ),
        "blocks": blocks,
    }


def _as_observed(
    ledger: SuccessorLedger, phase: Phase, row: dict[str, Any]
) -> dict[str, Any]:
    outcome = {
        "state": row["state"],
        "verdict": row["verdict"],
        "stop_reason": row.get("stop_reason"),
    }
    relative = row.get("evidence_path")
    if relative is None:
        return outcome
    path = ledger.root / phase / f"session-{row['ordinal']:04d}" / relative
    receipt = load_json_value(path)
    if receipt["session_id"] != row["session_id"]:
        raise ValueError("E2 cost receipt coordinates changed")
    outcome["receipt_sha256"] = sha256_file(path)
    if phase is Phase.ONLINE:
        if receipt["original_evaluator_verdict"] != row["verdict"]:
            raise ValueError("E2 original cost verdict changed")
        outcome["components"] = _components(receipt)
    else:
        if receipt["gpu_initialized"] is not False:
            raise ValueError("E2 offline cost receipt initialized the GPU")
        outcome["components"] = {
            field: receipt.get(field)
            for field in (
                "oracle_cpu_seconds",
                "wall_seconds",
                "committed",
                "available_observations",
                "allocated_upper_estimate_bytes",
            )
        }
    return outcome


def build_cost_report(ledger: SuccessorLedger) -> dict[str, Any]:
    """Read paired online and deferred outcomes from the same frozen campaign."""
    online = {
        row["session_id"]: _as_observed(ledger, Phase.ONLINE, row)
        for row in ledger.reconcile(Phase.ONLINE)
        if row["activity"] == Activity.COST
    }
    offline = {
        row["session_id"]: _as_observed(ledger, Phase.OFFLINE, row)
        for row in ledger.reconcile(Phase.OFFLINE)
    }
    return summarize_costs(online, offline)
