# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Verify exact-SHA RDNA4 evidence and its release-attestation binding."""

from __future__ import annotations

import argparse
from pathlib import Path

from sol_execbench.core.bench.performance_model.release import (
    DiagnosticReleaseAttestation,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_file,
)
from sol_execbench.core.platform.rdna4_validation import (
    HardwareValidationBinding,
    verify_validation_receipt,
)
from sol_execbench.core.scoring.release_packaging import ScoreReleaseAttestation


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument(
        "--attestation-kind",
        choices=("diagnostic", "score"),
        required=True,
    )
    parser.add_argument("--binding-output", type=Path, required=True)
    return parser.parse_args(argv)


def _load_binding(
    path: Path,
    kind: str,
) -> tuple[str, HardwareValidationBinding]:
    if kind == "diagnostic":
        attestation = load_json_file(DiagnosticReleaseAttestation, path)
    else:
        attestation = load_json_file(ScoreReleaseAttestation, path)
    return attestation.source_revision, attestation.hardware_validation


def main(argv: list[str] | None = None) -> int:
    """Verify the gate and emit its canonical nested release binding."""
    args = _parse_args(argv)
    binding = verify_validation_receipt(
        args.receipt,
        args.evidence_dir,
        expected_source_revision=args.source_revision,
    )
    source_revision, published_binding = _load_binding(
        args.attestation,
        args.attestation_kind,
    )
    if source_revision != args.source_revision:
        raise ValueError("release attestation source revision mismatch")
    if published_binding != binding:
        raise ValueError("release attestation hardware validation mismatch")
    atomic_write_json_value(
        args.binding_output,
        binding.model_dump(mode="json"),
    )
    print(args.binding_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
