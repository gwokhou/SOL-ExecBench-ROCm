# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Build a before/after transition ledger for the 55 custom-input readiness blockers.

Loads the fixed v1.34 before-baseline from ``out/rdna4-coverage-current/coverage.json``,
runs a fresh readiness classification, and records every transition or residual blocker
with precise classification.  The 235-problem denominator must remain stable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sol_execbench.core.dataset import (
    build_dataset_inventory,
    build_profiler_timing_coverage_report,
    classify_rocm_readiness,
    validate_categories,
)
from sol_execbench.core.checksums import stable_json_checksum

DEFAULT_DATASET_ROOT = Path("data/SOL-ExecBench/benchmark")
DEFAULT_BASELINE_PATH = Path("out/rdna4-coverage-current/coverage.json")
DEFAULT_OUTPUT_DIR = Path("out/rdna4-custom-input-transition-ledger")
DEFAULT_TIMING_EVIDENCE_DIR = Path("out/rdna4-timing-evidence/timing")
EXPECTED_PROBLEM_DENOMINATOR = 235

TRANSITION_SCHEMA_VERSION = "sol_execbench.custom_input_transition_ledger.v1"

RESIDUAL_CLASSES = frozenset(
    {
        "unsupported_custom_entrypoint",
        "gen_inputs_oom_blocked",
        "gen_inputs_schema_mismatch",
        "gen_inputs_device_mismatch",
        "gen_inputs_timeout",
        "execution_environment_unavailable",
        "custom_input_requires_evaluator_support",
    }
)


def _load_baseline(baseline_path: Path) -> dict:
    """Load and return the fixed before-baseline coverage JSON."""
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Baseline must be a JSON object: {baseline_path}")
    return payload


def _extract_custom_input_cohort(baseline: dict) -> list[dict]:
    """Return the list of problems with ``readiness_status == custom_input_blocked``."""
    return [
        p
        for p in baseline.get("problems", [])
        if p.get("readiness_status") == "custom_input_blocked"
    ]


def _build_after_lookup(
    readiness,
    report,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Build problem-level and workload-level lookup dicts keyed by problem_id."""
    problem_map: dict[str, dict] = {}
    for p in report.problems:
        problem_map[p.problem_id] = {
            "readiness_status": p.readiness_status,
            "status": p.status,
            "readiness_reason_codes": p.readiness_reason_codes,
            "readiness_blocker_types": p.readiness_blocker_types,
        }
    workload_map: dict[str, list[dict]] = {}
    for w in readiness.workloads:
        entry = {
            "workload_uuid": w.workload_uuid,
            "row_index": w.row_index,
            "status": w.status,
            "reason_codes": [r.code for r in w.reasons],
        }
        workload_map.setdefault(w.problem_id, []).append(entry)
    return problem_map, workload_map


def _classify_transition(
    before_status: str, after_status: str, after_reason_codes: list[str]
) -> tuple[str, str | None]:
    """Determine transition string and optional residual class.

    Returns (transition, residual_class) where residual_class is populated
    only when the problem remains blocked.
    """
    if before_status != "custom_input_blocked":
        return "no_change", None

    if after_status == "ready":
        return "promoted_to_ready", None
    if after_status == "timing_fallback":
        return "promoted_to_timing_fallback", None
    if after_status == "profiler_backed":
        return "promoted_to_profiler_backed", None
    if after_status == "reference_oom_blocked":
        return "transitioned_to_oom_blocked", None
    if after_status == "profiler_blocked":
        return "transitioned_to_profiler_blocked", None
    if after_status == "runtime_blocked":
        return "transitioned_to_runtime_blocked", None
    if after_status == "custom_input_blocked":
        # Still blocked for the same reason
        residual = _pick_residual_class(after_reason_codes)
        return "no_change", residual

    if after_status == "readiness_blocked":
        residual = _pick_residual_class(after_reason_codes)
        return "residual_readiness_blocked", residual

    if after_status in (
        "schema_input_blocked",
        "dtype_blocked",
        "unsupported_nvidia_only_path",
        "needs_hardware_evidence",
    ):
        return f"transitioned_to_{after_status}", None

    # Any other blocked status
    return "residual_readiness_blocked", None


def _pick_residual_class(reason_codes: list[str]) -> str | None:
    """Pick the most specific residual class from reason codes."""
    for code in reason_codes:
        if code in RESIDUAL_CLASSES:
            return code
    # If no specific code matches, return a generic marker
    return "execution_environment_unavailable" if reason_codes else None


def build_transition_ledger(
    *,
    dataset_root: Path,
    baseline_path: Path,
    output_dir: Path,
    timing_evidence_dirs: tuple[Path, ...] = (DEFAULT_TIMING_EVIDENCE_DIR,),
) -> dict:
    """Build and return the transition ledger payload."""
    # D-01: Load fixed before-baseline
    baseline = _load_baseline(baseline_path)
    baseline_checksum = (
        baseline.get("coverage_checksum", {}).get("value")
        if isinstance(baseline.get("coverage_checksum"), dict)
        else None
    )

    # D-02: Record baseline checksum and path
    cohort = _extract_custom_input_cohort(baseline)
    if not cohort:
        raise ValueError("No custom_input_blocked problems found in baseline")

    # D-03: Record baseline path as supplementary context
    baseline_path_str = baseline_path.as_posix()

    # Step 3: Fresh readiness classification
    selected_categories = validate_categories(None)
    inventory = build_dataset_inventory(dataset_root, categories=selected_categories)
    readiness = classify_rocm_readiness(inventory, dataset_root=dataset_root)
    report = build_profiler_timing_coverage_report(
        readiness,
        dataset_root=dataset_root,
        timing_evidence_dirs=timing_evidence_dirs,
        expected_problem_denominator=EXPECTED_PROBLEM_DENOMINATOR,
    )

    # Denominator assertion
    denominator = report.totals.problem_denominator
    denominator_ok = denominator == EXPECTED_PROBLEM_DENOMINATOR

    # Build lookups
    after_problems, after_workloads = _build_after_lookup(readiness, report)

    # Step 4: Build transition records
    transitions = []
    for before in cohort:
        problem_id = before["problem_id"]
        after_info = after_problems.get(problem_id)
        if after_info is None:
            # Problem vanished -- unexpected
            transitions.append(
                {
                    "problem_id": problem_id,
                    "category": before.get("category", ""),
                    "before_readiness_status": "custom_input_blocked",
                    "before_reason_codes": before.get("readiness_reason_codes", []),
                    "after_readiness_status": None,
                    "after_reason_codes": [],
                    "transition": "residual_readiness_blocked",
                    "residual_class": "execution_environment_unavailable",
                    "workload_transitions_available": False,
                    "workload_transitions": [],
                }
            )
            continue

        after_status = after_info["readiness_status"]
        after_reason_codes = after_info.get("readiness_reason_codes", [])
        transition, residual_class = _classify_transition(
            "custom_input_blocked",
            after_status,
            after_reason_codes,
        )

        # Step 5: Workload-level transitions
        wl_after = after_workloads.get(problem_id, [])
        wl_available = len(wl_after) > 0
        wl_transitions = []
        if wl_available:
            for w in wl_after:
                wl_transitions.append(
                    {
                        "workload_uuid": w["workload_uuid"],
                        "row_index": w["row_index"],
                        "after_status": w["status"],
                        "after_reason_codes": w["reason_codes"],
                    }
                )

        transitions.append(
            {
                "problem_id": problem_id,
                "category": before.get("category", ""),
                "before_readiness_status": "custom_input_blocked",
                "before_reason_codes": before.get("readiness_reason_codes", []),
                "after_readiness_status": after_status,
                "after_reason_codes": after_reason_codes,
                "transition": transition,
                "residual_class": residual_class,
                "workload_transitions_available": wl_available,
                "workload_transitions": wl_transitions if wl_available else [],
            }
        )

    # Build transition type counts
    transition_counts = Counter(t["transition"] for t in transitions)
    residual_counts = Counter(
        t["residual_class"] for t in transitions if t["residual_class"] is not None
    )

    ledger = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "baseline_path": baseline_path_str,
        "baseline_checksum": baseline_checksum,
        "cohort_size": len(cohort),
        "denominator_assertion": {
            "expected": EXPECTED_PROBLEM_DENOMINATOR,
            "actual": denominator,
            "passed": denominator_ok,
        },
        "transition_counts": dict(sorted(transition_counts.items())),
        "residual_class_counts": dict(sorted(residual_counts.items())),
        "transitions": sorted(transitions, key=lambda t: t["problem_id"]),
    }
    # Stable checksum
    ledger["ledger_checksum"] = stable_json_checksum(
        {**ledger, "ledger_checksum": None}
    )
    return ledger


def render_transition_summary(ledger: dict) -> str:
    """Render a human-readable transition summary."""
    lines = [
        "# Custom Input Transition Summary",
        "",
        f"- Baseline: `{ledger['baseline_path']}`",
        f"- Baseline checksum: `{ledger['baseline_checksum']}`",
        f"- Cohort size: {ledger['cohort_size']}",
        f"- Denominator assertion: expected={ledger['denominator_assertion']['expected']}, "
        f"actual={ledger['denominator_assertion']['passed']}",
        "",
        "## Transition Counts",
        "",
        "| Transition | Count |",
        "| --- | ---: |",
    ]
    for transition, count in sorted(ledger["transition_counts"].items()):
        lines.append(f"| {transition} | {count} |")

    if ledger["residual_class_counts"]:
        lines.extend(
            [
                "",
                "## Residual Blocker Classes",
                "",
                "| Residual Class | Count |",
                "| --- | ---: |",
            ]
        )
        for cls, count in sorted(ledger["residual_class_counts"].items()):
            lines.append(f"| {cls} | {count} |")

    # List problems per transition type
    lines.extend(
        [
            "",
            "## Problem Transitions",
            "",
        ]
    )
    for t in ledger["transitions"]:
        residual = f" → `{t['residual_class']}`" if t.get("residual_class") else ""
        wl_marker = (
            " (workload evidence available)"
            if t.get("workload_transitions_available")
            else " (workload_transition_unavailable)"
        )
        lines.append(
            f"- `{t['problem_id']}`: {t['before_readiness_status']} → "
            f"{t['after_readiness_status']} ({t['transition']}){residual}{wl_marker}"
        )

    lines.extend(
        [
            "",
            "## Readiness Movement Disclaimer (D-09)",
            "",
            "Readiness or smoke movement is **not** full validation success.  Promoted "
            "problems are ready to attempt execution; they have not necessarily passed "
            "correctness or performance validation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Fixed before-baseline coverage JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for transition artifacts.",
    )
    parser.add_argument(
        "--timing-evidence-dir",
        action="append",
        type=Path,
        default=None,
        help="Timing evidence directory; may be specified multiple times.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    timing_dirs = (
        tuple(args.timing_evidence_dir)
        if args.timing_evidence_dir
        else (DEFAULT_TIMING_EVIDENCE_DIR,)
    )
    try:
        ledger = build_transition_ledger(
            dataset_root=args.dataset_root,
            baseline_path=args.baseline_path,
            output_dir=args.output_dir,
            timing_evidence_dirs=timing_dirs,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    # Denominator gate
    if not ledger["denominator_assertion"]["passed"]:
        print(
            f"ERROR: denominator assertion failed: "
            f"expected {ledger['denominator_assertion']['expected']}, "
            f"got {ledger['denominator_assertion']['actual']}"
        )
        return 1

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "transition-ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "transition-summary.md").write_text(
        render_transition_summary(ledger),
        encoding="utf-8",
    )

    # Report
    print(f"Cohort: {ledger['cohort_size']} problems")
    for transition, count in sorted(ledger["transition_counts"].items()):
        print(f"  {transition}: {count}")
    print(f"Denominator: {ledger['denominator_assertion']['actual']} (OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
