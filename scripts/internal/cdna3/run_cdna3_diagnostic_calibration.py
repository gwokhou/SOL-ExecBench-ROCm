#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Collect gfx942 (CDNA3 / MI300X) MFMA peak engineering evidence.

This is a focused non-formal companion to the gfx1200 diagnostic calibration.
It compiles and runs the CDNA3 MFMA matrix probes on a real gfx942 device and
records peak throughput plus the emitted ``V_MFMA_*`` instruction evidence.

Results are engineering/inexact: the HANDSOFF P1 MI300X capacity policy and a
formal resource-peak calibration receipt remain separate hardware-validated
deliverables. Deferred to on-device iteration: the fp8 MFMA probe (FNUZ
encoding) and the int8 MFMA probe (32x32x8 output layout), plus the VRAM
working-set derivation which needs a wave64 ``diagnostic_microarchitecture``
probe variant.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.platform.amdgpu_code_object import extract_code_object
from sol_execbench.core.platform.isa_validation import analyze_isa_disassembly
from sol_execbench.core.platform.runtime import (
    detect_rocm_device,
    resolve_rocm_tool,
)
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.timestamps import utc_timestamp

_PROBES_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "sol_execbench"
    / "data"
    / "hardware_calibration_probes"
)
# (source filename, expected MFMA ISA mnemonic). The exact disassembly
# mnemonics MUST be confirmed on gfx942 hardware; a mismatch fails closed.
_MFMA_PROBES = (
    ("matrix_bf16_bf16_mfma.hip", "V_MFMA_F32_16X16X16BF16"),
    ("matrix_fp16_fp16_mfma.hip", "V_MFMA_F32_16X16X16F16"),
)
COMMAND_TIMEOUT_SECONDS = 180.0


def _compile_probe(
    hipcc: Path,
    source: Path,
    architecture: str,
    output: Path,
) -> None:
    command = [
        str(hipcc),
        str(source),
        "-O3",
        "-std=c++17",
        f"--offload-arch={architecture}",
        "-o",
        str(output),
    ]
    completed = run_in_process_group_bounded(
        command,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"probe compilation failed: {completed.stderr}")


def _run_probe(binary: Path) -> float:
    completed = run_in_process_group_bounded(
        [str(binary)],
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"probe failed: {completed.stderr}")
    peaks = [
        float(line.removeprefix("RESULT ").strip())
        for line in completed.stdout.splitlines()
        if line.startswith("RESULT ")
    ]
    if not peaks:
        raise RuntimeError("probe emitted no RESULT lines")
    return max(peaks)


def run_calibration(*, output: Path) -> Path:
    """Compile and run the gfx942 MFMA probes, then write a summary receipt."""
    hipcc = resolve_rocm_tool("hipcc")
    if hipcc is None:
        raise RuntimeError("hipcc is unavailable")
    device = detect_rocm_device()
    if device.gfx_target != "gfx942":
        raise RuntimeError(
            f"cdna3 diagnostic calibration requires gfx942, got "
            f"{device.gfx_target}"
        )
    probes: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as workspace_dir:
        workspace = Path(workspace_dir)
        for source_name, expected_instruction in _MFMA_PROBES:
            source = _PROBES_DIR / source_name
            binary = workspace / source.with_suffix("").name
            _compile_probe(hipcc, source, device.gfx_target, binary)
            peak_tflops = _run_probe(binary)
            extracted = extract_code_object(
                binary,
                device.gfx_target,
                workspace,
                timeout_seconds=COMMAND_TIMEOUT_SECONDS,
            )
            analysis = analyze_isa_disassembly(
                device.gfx_target,
                extracted.disassembly,
                expected_instructions=(expected_instruction,),
            )
            if not analysis.matched_instruction_counts.get(
                expected_instruction
            ):
                raise RuntimeError(
                    f"{source_name} did not emit {expected_instruction}"
                )
            probes.append(
                {
                    "source": source_name,
                    "expected_instruction": expected_instruction,
                    "peak_tflops": peak_tflops,
                    "binary_sha256": sha256_file(binary),
                    "code_object_sha256": extracted.sha256,
                    "disassembly_sha256": extracted.disassembly_sha256,
                    "decoded_instruction_count": (
                        analysis.decoded_instruction_count
                    ),
                    "matched_instruction_counts": dict(
                        analysis.matched_instruction_counts,
                    ),
                    "spec_provenance": analysis.provenance.to_dict(),
                },
            )
    payload = {
        "confidence": "inexact",
        "gfx_target": device.gfx_target,
        "device_name": device.name,
        "rocm_hip_version": device.hip_version,
        "created_at": utc_timestamp(),
        "probe_summary_sha256": stable_json_checksum(probes),
        "probes": probes,
    }
    atomic_write_json_value(output, payload)
    return output


def main() -> int:
    """Run the gfx942 MFMA peak calibration and print the receipt path."""
    parser = argparse.ArgumentParser(
        description="Collect gfx942 MFMA peak engineering evidence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON receipt output path.",
    )
    args = parser.parse_args()
    receipt = run_calibration(output=args.output)
    print(json.dumps({"status": "ok", "receipt": str(receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
