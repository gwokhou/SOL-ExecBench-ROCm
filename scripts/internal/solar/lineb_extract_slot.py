"""One extraction slot for the lineB equality campaign (worker mode)."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def main() -> int:
    """Extract one workload and record the operator-graph digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="tree whose corpus and code to use (default: this script's tree)",
    )
    args = parser.parse_args()
    repo = (args.repo or REPO).resolve()
    import hashlib

    from sol_execbench.core.solar_bridge.workload_context import (
        load_solar_workload_context,
        solar_conversion_request,
    )
    from solar.graph.extraction import extract_operator_graph
    from solar.ir.contracts import IRPath

    problem, workload_uuid = args.item.rsplit("/", 1)
    context = load_solar_workload_context(
        repo / "problems" / "AMD_AKA" / problem, workload_uuid, "cpu"
    )
    conversion = solar_conversion_request(context, "cpu", IRPath.MAKE_FX_ATEN)
    selected = IRPath.MAKE_FX_ATEN
    try:
        artifact = extract_operator_graph(
            conversion.reference,
            tuple(conversion.input_factory(conversion.trace_seed)),
            device="cpu",
            output_dir=args.output,
            name=conversion.analysis_id,
            extraction_kind=selected.extraction_kind,
        )
    except Exception as exc:  # noqa: BLE001 -- record the rejection
        args.output.joinpath("graph.sha").write_text("")
        args.output.joinpath("error.txt").write_text(
            f"{type(exc).__name__}: {exc}"[:2048]
        )
        return 0
    digest = hashlib.sha256(artifact.path.read_bytes()).hexdigest()
    args.output.joinpath("graph.sha").write_text(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
