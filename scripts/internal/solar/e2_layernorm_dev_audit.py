"""Audit the frozen development denominator after the GPU has been handed back."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.internal.solar.e2_archive_arrays import read_output_arrays
from scripts.internal.solar.e2_layernorm_dev_gpu import (
    ACTIVITY_DIRECTORY,
    COMPLETE_CALLS,
    CORRECTNESS_CALLS,
    Slot,
    roster,
    verify_bundle,
)
from scripts.internal.solar.e2_layernorm_dev_inputs import rotate_inputs
from scripts.internal.solar.e2_postprocess import decode_observation
from sol_execbench.core.bench.custom_inputs import derive_custom_input_seed
from sol_execbench.core.bench.judge.diagnostics import (
    original_numeric_check,
    stronger_check,
)
from sol_execbench.core.bench.judge.inputs import (
    JudgeInputs,
    generate_seeded_inputs,
)
from sol_execbench.core.bench.judge.storage import tensor_identity
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_file


def _call_coordinates(rows: list[dict[str, Any]]) -> None:
    legal_order = [
        *(
            ("correctness", round_index)
            for round_index in range(CORRECTNESS_CALLS)
        ),
        ("timing_iteration", CORRECTNESS_CALLS - 1),
    ]
    if [(r["operation"], r["round_index"]) for r in rows] != legal_order[
        : len(rows)
    ]:
        raise ValueError(
            "recorded calls are not a prefix of the frozen schedule"
        )
    if len(rows) == COMPLETE_CALLS and (
        rows[-1]["trial_index"],
        rows[-1]["iteration_index"],
    ) != (0, 0):
        raise ValueError("recorded timed input has the wrong coordinates")


def _call(
    slot: Slot,
    definition: Definition,
    workload: Workload,
    archive: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    if sha256_file(archive) != row["artifact_sha256"]:
        raise ValueError("raw archive identity changed")
    actual = decode_observation(archive, row)
    timing = row["operation"] == "timing_iteration"
    expected = _expected_input(slot, definition, workload, row, timing=timing)
    case = slot.session.case
    rotated = rotate_inputs(expected, slot.shift)
    digest = tensor_identity(actual.inputs.values)
    if (
        digest != row["input_sha256"]
        or digest != tensor_identity(rotated.values)
        or actual.inputs.construction != rotated.construction
    ):
        raise ValueError("actual features or permutation did not reproduce")
    with np.load(archive, allow_pickle=False) as arrays:
        value = read_output_arrays(arrays, with_oracle=True).get(
            "oracle_rounded"
        )
    oracle = (
        torch.tensor(value, dtype=torch.float32) if value is not None else None
    )
    numeric, margin = original_numeric_check(
        definition, workload, actual.inputs, actual.reference, actual.actual
    )
    gate = (
        "incumbent_timing_validation_passed"
        if timing
        else "incumbent_numeric_passed"
    )
    if numeric != row[gate]:
        raise ValueError(
            "retained output disagrees with the original numeric gate"
        )
    return {
        "archive": archive.name,
        "archive_sha256": row["artifact_sha256"],
        "input_sha256": digest,
        "unrotated_input_sha256": tensor_identity(expected.values),
        "unrotated_oracle_sha256": (
            tensor_identity((oracle.roll(-slot.shift, dims=-1),))
            if oracle is not None
            else None
        ),
        "unrotated_candidate_sha256": tensor_identity(
            (actual.actual.roll(-slot.shift, dims=-1),)
        ),
        "unrotated_reference_sha256": tensor_identity(
            (actual.reference.roll(-slot.shift, dims=-1),)
        ),
        "candidate_4ulp": (
            stronger_check(case, actual.inputs, actual.actual, oracle)
            if oracle is not None
            else None
        ),
        "reference_4ulp": (
            stronger_check(case, actual.inputs, actual.reference, oracle)
            if oracle is not None
            else None
        ),
        "numeric_passed": numeric,
        "minimum_margin": margin,
        "operation": row["operation"],
        "witness_certified": row.get("witness_certified"),
    }


def _expected_input(
    slot: Slot,
    definition: Definition,
    workload: Workload,
    row: dict[str, Any],
    *,
    timing: bool,
) -> JudgeInputs:
    seed = derive_custom_input_seed(
        definition,
        workload,
        row_index=row["row_index"],
        base_seed=slot.session.case.require_development(),
        round_index=row["round_index"],
        run_nonce=slot.nonce,
        variation_index=0 if timing else None,
    )
    case = slot.session.case
    return generate_seeded_inputs(
        case.task,
        case.family,
        definition.get_resolved_axes_values(workload.axes),
        seed,
    )


def _review(
    slot: Slot, folder: Path, identities: dict[str, Any]
) -> dict[str, Any]:
    receipt = folder / "session.json"
    if not receipt.exists():
        return {
            "state": "interrupted" if folder.exists() else "not_reached",
            "calls": [],
        }
    result = load_json_value(receipt)
    case = slot.session.case
    bound = identities[
        f"{case.workload_index}/{case.family}/{slot.session.strategy}"
    ]
    if (
        result["session_id"] != slot.session.identity
        or result["feature_shift"] != slot.shift
        or result["candidate_sha256"] != bound["candidate_sha256"]
        or result["reference_sha256"] != bound["reference_sha256"]
        or sha256_file(folder / "candidate.py") != bound["candidate_sha256"]
        or result["original_checks"] != bound["checks"]
    ):
        raise ValueError("staged slot identity disagrees with the freeze")
    observed = result.get("observations", {})
    rows = observed.get("observations", [])
    if observed and (
        observed["private_nonce_after_channel_close"] != slot.nonce
        or observed["publication_after_candidate_exit"] is not True
    ):
        raise ValueError(
            "observation publication or development nonce failed verification"
        )
    _call_coordinates(rows)
    definition = Definition.model_validate_json(
        (folder / "definition.json").read_text()
    )
    workload = Workload.model_validate_json(
        (folder / "workload.jsonl").read_text()
    )
    if workload.uuid != slot.uuid:
        raise ValueError("the staged workload UUID changed")
    files = folder / "judge-observations"
    calls = [
        _call(slot, definition, workload, files / f"call-{i:03d}.npz", row)
        for i, row in enumerate(rows)
    ]
    if len({c["input_sha256"] for c in calls}) != len(calls):
        raise ValueError("an input repeated within the isolated session")
    return {
        "state": _slot_state(slot, result, calls),
        "verdict": result["original_evaluator_verdict"],
        "stop_reason": result["stop_reason"],
        "session_sha256": sha256_file(receipt),
        "calls": calls,
    }


def _slot_state(
    slot: Slot, result: dict[str, Any], calls: list[dict[str, Any]]
) -> str:
    if result["stop_reason"] or any(c["reference_4ulp"] is None for c in calls):
        return "indeterminate"
    verdict = result["original_evaluator_verdict"]
    if slot.lane == "control":
        passed = (
            verdict == "PASSED"
            and len(calls) == COMPLETE_CALLS
            and all(
                c["numeric_passed"]
                and c["candidate_4ulp"]
                and c["reference_4ulp"]
                for c in calls
            )
        )
    else:
        passed = (
            verdict == "INCORRECT_NUMERICAL"
            and bool(calls)
            and calls[-1]["operation"] == "correctness"
            and calls[-1]["numeric_passed"] is False
            and calls[-1]["witness_certified"] is True
            and calls[-1]["candidate_4ulp"] is False
            and calls[-1]["reference_4ulp"] is True
        )
    return "passed" if passed else "failed"


def _columns(rows: list[dict[str, Any]], field: str) -> list[tuple[Any, ...]]:
    return [tuple(c[field] for c in row["calls"]) for row in rows]


def _group_state(rows: list[dict[str, Any]], *, permuted: bool) -> str:
    states = {row["state"] for row in rows}
    if "failed" in states:
        return "failed"
    if states != {"passed"}:
        return "indeterminate"
    equal = (
        ("unrotated_input_sha256", "unrotated_oracle_sha256")
        if permuted
        else (
            "input_sha256",
            "unrotated_candidate_sha256",
            "unrotated_reference_sha256",
        )
    )
    if not all(len(set(_columns(rows, field))) == 1 for field in equal):
        return "failed"
    if permuted and any(
        len(set(hashes)) != len(rows)
        for hashes in zip(*_columns(rows, "input_sha256"), strict=True)
    ):
        return "failed"
    return "passed"


def _comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed, varied = rows[:3], rows[2:]
    permuted = _group_state(varied, permuted=True)
    return {
        "fixed": _group_state(fixed, permuted=False),
        "permuted": permuted,
        **{
            f"permuted_{side}_exact_after_inverse": (
                len(set(_columns(varied, f"unrotated_{side}_sha256"))) == 1
                if all(len(row["calls"]) == COMPLETE_CALLS for row in varied)
                else None
            )
            for side in ("candidate", "reference")
        },
    }


def audit(
    bundle: Path, spec: dict[str, Any], expected_sha: str
) -> dict[str, Any]:
    """Keep every incomplete, failed and uncertain slot in the terminal report."""
    work = bundle / ACTIVITY_DIRECTORY
    slots = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slot in roster():
        try:
            review = _review(slot, work / slot.name, spec["staged_identities"])
        except (KeyError, OSError, TypeError, ValueError) as error:
            review = {
                "state": "invalid_evidence",
                "detail": str(error),
                "calls": [],
            }
        row = {**slot.metadata(), **review}
        slots.append(row)
        if slot.lane == "control":
            groups[
                f"{row['workload_index']}/{row['family']}/{row['strategy']}"
            ].append(row)
    paired = {key: _comparisons(rows) for key, rows in groups.items()}
    negative = {
        strategy: dict(
            Counter(r["state"] for r in slots if r["strategy"] == strategy)
        )
        for strategy in ("p1", "p2")
    }
    fixed = Counter(r["fixed"] for r in paired.values())
    permuted = Counter(r["permuted"] for r in paired.values())
    handback_path = work / "gpu-closeout.json"
    handback = load_json_value(handback_path) if handback_path.exists() else {}
    complete = (
        fixed == permuted == {"passed": 24}
        and all(counts == {"passed": 18} for counts in negative.values())
        and handback.get("clean") is True
        and handback.get("returncode") == 0
    )
    return {
        "freeze_sha256": expected_sha,
        "performance_authority": False,
        "acceptance_passed": complete,
        "slots": slots,
        "groups": paired,
        "slot_states": dict(Counter(r["state"] for r in slots)),
        "fixed_group_states": dict(fixed),
        "permuted_group_states": dict(permuted),
        "negative_states": negative,
        "gpu_handback_clean": handback.get("clean"),
        "gpu_handback_sha256": sha256_file(handback_path) if handback else None,
    }


def main() -> None:
    """Verify frozen evidence in the device-free runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    if Path("/dev/kfd").exists() or any(Path("/dev/dri").glob("renderD*")):
        raise ValueError("LayerNorm GPU audit requires a device-free container")
    spec = verify_bundle(args.repository, args.bundle, args.expected_sha)
    result = audit(args.bundle, spec, args.expected_sha)
    output = args.bundle / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    atomic_write_json_value(output, result)


if __name__ == "__main__":
    main()
