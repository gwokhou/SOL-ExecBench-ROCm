# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure RX 9060 XT (gfx1200) resource peaks under locked clocks.

The packaged RX_9060_XT profile combines published matrix throughput with
derived non-matrix ceilings. This script measures the *sustained* peak of every
non-exempt precision/resource calibration target under locked clocks, verifies
the relevant VALU/WMMA instruction through the machine-readable ISA, emitted
code object, and runtime result, and emits a hash-bound audit artifact.

SOL bound discipline (see docs/SCORING-V3.md): the published bound uses the
AMD *nominal* theoretical peak (the fastest the silicon could go), not the
sustained probe result. The empirical measurement is therefore audit EVIDENCE
(it proves the hardware is real, clocks were locked, and sustained <= nominal);
it must never replace the nominal rate in resource_limits. A measured value
exceeding nominal is a contradiction and fails calibration.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.platform.amdgpu_code_object import extract_code_object
from sol_execbench.core.platform.environment import collect_pytorch_rocm_summary
from sol_execbench.core.platform.isa_validation import (
    ISAInstructionRequirement,
    analyze_isa_disassembly,
    inspect_isa_requirements,
)
from sol_execbench.core.platform.runtime import resolve_rocm_tool
from sol_execbench.core.process.logs import redacted_text_tail
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from solar.rocm.architecture import (
    RESOURCE_PEAK_CALIBRATION_SCHEMA_VERSION,
    RESOURCE_PEAK_TIMING_PROFILE,
    resource_peak_payload_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SOLAR_ROOT = REPO_ROOT / "src" / "solar"
PROBE_DIR = (
    REPO_ROOT / "src" / "sol_execbench" / "data" / "hardware_calibration_probes"
)

SCHEMA_VERSION = RESOURCE_PEAK_CALIBRATION_SCHEMA_VERSION
TIMING_PROFILE = RESOURCE_PEAK_TIMING_PROFILE
COMMAND_TIMEOUT_SECONDS = 120.0
MAX_CAPTURE_BYTES = 1024 * 1024
SAMPLES_PER_PROCESS_BATCH = 7
MIN_HELD_OUT_PROCESS_BATCHES = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_725


@dataclass(frozen=True)
class TuningSpec:
    """One bounded compile-time search space for an instruction probe."""

    parameter: str
    compiler_macro: str
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class SampleBatch:
    """Raw device-event samples from one fresh probe process."""

    process_batch: int
    samples: tuple[float, ...]
    telemetry_before: dict[str, Any] | None = None
    telemetry_after: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible sample batch."""
        payload: dict[str, Any] = {
            "process_batch": self.process_batch,
            "samples": list(self.samples),
            "median": statistics.median(self.samples),
        }
        if self.telemetry_before is not None:
            payload["telemetry_before"] = self.telemetry_before
        if self.telemetry_after is not None:
            payload["telemetry_after"] = self.telemetry_after
        return payload


@dataclass(frozen=True)
class PreparedProbe:
    """A probe whose tuning configuration and executable are frozen."""

    specification: Mapping[str, Any]
    source: Path
    binary: Path
    compiler_defines: Mapping[str, int]
    selected_configuration: Mapping[str, int]
    tuning_evidence: Mapping[str, Any]


FP32_TUNING = TuningSpec(
    parameter="accumulator_count",
    compiler_macro="SOL_FP32_ACCUMULATOR_COUNT",
    candidates=(2, 4, 8, 16),
)
WMMA_TUNING = TuningSpec(
    parameter="waves_per_reported_wgp",
    compiler_macro="SOL_WMMA_WAVES_PER_WGP",
    candidates=(8, 16, 32, 64, 128),
)

# These must match the non-exempt declarations in RX_9060_XT.yaml. The packaged
# profile loader independently compares its declarations with the signed audit,
# so omitting a target here cannot silently weaken formal analysis.
REQUIRED_PRECISIONS = ("bf16", "fp16", "fp32", "fp8", "int8")
REQUIRED_RESOURCE_MODES = (
    "mfma/bf16->fp32",
    "mfma/fp16->fp32",
    "mfma/fp8->fp32",
    "mfma/int8->int32",
    "reduction/generic",
    "sfu/generic",
    "valu/bf16",
    "valu/fp16",
    "valu/fp32",
)
REQUIRED_INSTRUCTION_CHECKS = (
    "bf16_valu_fallback",
    "bf16_wmma",
    "fp16_packed_valu_fma",
    "fp16_wmma",
    "fp32_valu_fma",
    "fp8_wmma",
    "int8_wmma",
)

# Each probe maps its runtime measurement onto explicit precision/resource
# targets. Compute probes print TOP/s or TFLOP/s and therefore scale by 1e12;
# the stream probe prints GB/s and scales by 1e9 bytes/s.
PROBES: tuple[dict[str, Any], ...] = (
    {
        "source": "vector_fp32_fp32.hip",
        "resource": "valu",
        "mode": "fp32",
        "covers_precisions": ("fp32",),
        "covers_resource_modes": ("valu/fp32",),
        "ops_per_result": 1e12,
        "nominal_ops_per_second": 25_600_000_000_000.0,
        "nominal_source": (
            "amd.com RX 9060 XT spec; FP32 Vector via two independent "
            "V_DUAL_FMAC_F32 operations"
        ),
        "tracked_instructions": ("V_DUAL_FMAC_F32",),
        "tuning": FP32_TUNING,
    },
    {
        "source": "vector_fp16_fp16.hip",
        "resource": "valu",
        "mode": "fp16",
        "covers_precisions": (),
        "covers_resource_modes": ("valu/fp16",),
        "ops_per_result": 1e12,
        "nominal_ops_per_second": 25_600_000_000_000.0,
        "nominal_source": "amd.com RX 9060 XT spec; FP16 Vector",
        "tracked_instructions": ("V_PK_FMA_F16", "V_PK_FMAC_F16"),
    },
    {
        "source": "vector_bf16_bf16.hip",
        "resource": "valu",
        "mode": "bf16",
        "covers_precisions": (),
        "covers_resource_modes": ("valu/bf16",),
        "ops_per_result": 1e12,
        # gfx1200 VALU has no native packed bf16 FMA.  This probe measures the
        # compiler's FP32 arithmetic plus BF16 conversion/rounding fallback;
        # AMD publishes no BF16 Vector peak, so no efficiency ratio is valid.
        "nominal_ops_per_second": None,
        "nominal_source": (
            "diagnostic fallback throughput; AMD publishes no RX 9060 XT "
            "BF16 Vector peak"
        ),
        "tracked_instructions": (
            "V_PK_FMA_BF16",
            "V_PK_FMAC_BF16",
            "V_FMA_F32",
            "V_FMAAK_F32",
        ),
    },
    {
        "source": "matrix_fp16_fp16_wmma.hip",
        "resource": "mfma",
        "mode": "fp16->fp32",
        "covers_precisions": ("fp16",),
        "covers_resource_modes": ("mfma/fp16->fp32",),
        "ops_per_result": 1e12,
        "nominal_ops_per_second": 103_000_000_000_000.0,
        "nominal_source": "amd.com RX 9060 XT spec; FP16 matrix throughput",
        "tracked_instructions": ("V_WMMA_F32_16X16X16_F16",),
        "tuning": WMMA_TUNING,
    },
    {
        "source": "matrix_bf16_bf16_wmma.hip",
        "resource": "mfma",
        "mode": "bf16->fp32",
        "covers_precisions": ("bf16",),
        "covers_resource_modes": ("mfma/bf16->fp32",),
        "ops_per_result": 1e12,
        "nominal_ops_per_second": 103_000_000_000_000.0,
        "nominal_source": (
            "derived from FP16/BF16 gfx12 WMMA throughput parity; AMD does not "
            "publish an RX 9060 XT BF16 Matrix figure"
        ),
        "tracked_instructions": ("V_WMMA_F32_16X16X16_BF16",),
        "tuning": WMMA_TUNING,
    },
    {
        "source": "matrix_fp8_fp8_wmma.hip",
        "resource": "mfma",
        "mode": "fp8->fp32",
        "covers_precisions": ("fp8",),
        "covers_resource_modes": ("mfma/fp8->fp32",),
        "ops_per_result": 1e12,
        "nominal_ops_per_second": 205_000_000_000_000.0,
        "nominal_source": "amd.com RX 9060 XT spec; FP8 matrix throughput",
        "tracked_instructions": ("V_WMMA_F32_16X16X16_FP8_FP8",),
        "tuning": WMMA_TUNING,
    },
    {
        "source": "matrix_int8_int8_wmma.hip",
        "resource": "mfma",
        "mode": "int8->int32",
        "covers_precisions": ("int8",),
        "covers_resource_modes": ("mfma/int8->int32",),
        "ops_per_result": 1e12,
        "nominal_ops_per_second": 205_000_000_000_000.0,
        "nominal_source": "amd.com RX 9060 XT spec; INT8 matrix throughput",
        "tracked_instructions": ("V_WMMA_I32_16X16X16_IU8",),
        "tuning": WMMA_TUNING,
    },
    {
        "source": "transcendental_fp32_fp32.hip",
        "resource": "sfu",
        "mode": "generic",
        "covers_precisions": (),
        "covers_resource_modes": ("sfu/generic",),
        "ops_per_result": 1e12,
        # SFU transcendentals issue at a fraction of the VALU rate; the nominal
        # is left equal to the loose ceiling and the measurement is evidence.
        "nominal_ops_per_second": None,
        "nominal_source": "RDNA4 SFU issue rate is workload-specific; not separately published",
    },
    {
        "source": "reduction_fp32_fp32.hip",
        "resource": "reduction",
        "mode": "generic",
        "covers_precisions": (),
        "covers_resource_modes": ("reduction/generic",),
        "ops_per_result": 1e12,
        "nominal_ops_per_second": None,
        "nominal_source": "reduction is memory-bound; nominal tracked via bandwidth, not a fixed compute rate",
    },
    {
        "source": "stream_copy_fp32_fp32.hip",
        "resource": "bandwidth",
        "mode": "fp32",
        "covers_precisions": (),
        "covers_resource_modes": (),
        "ops_per_result": 1e9,
        "nominal_ops_per_second": 320_000_000_000.0,
        "nominal_source": "amd.com RX 9060 XT spec; memory bandwidth (320 GB/s)",
    },
)


def _run(
    cmd: list[str],
    *,
    timeout_seconds: float = COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    completed = run_in_process_group_bounded(
        cmd,
        timeout=timeout_seconds,
        max_capture_bytes=MAX_CAPTURE_BYTES,
    )
    if completed.returncode != 0:
        detail = redacted_text_tail(completed.stderr or completed.stdout or "")
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {detail}",
        )
    return completed


def _required_rocm_tool(name: str) -> str:
    path = resolve_rocm_tool(name)
    if path is None:
        raise FileNotFoundError(f"required ROCm tool is unavailable: {name}")
    return str(path)


def _git_revision() -> str:
    try:
        return _run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- revision metadata is best-effort
        return "unknown"


def _device_identity() -> dict[str, Any]:
    summary = collect_pytorch_rocm_summary()
    if (
        not summary.available
        or not summary.device_name
        or not summary.gfx_target
    ):
        raise RuntimeError(
            "PyTorch ROCm could not identify a visible AMD GPU: "
            f"{summary.error or 'device metadata unavailable'}",
        )
    amdsmi = _run([_required_rocm_tool("amd-smi"), "version"]).stdout
    hipcc_result = _run([_required_rocm_tool("hipcc"), "--version"])
    hipcc_output = f"{hipcc_result.stdout}\n{hipcc_result.stderr}"
    return {
        "device_name": summary.device_name,
        "gfx_target": summary.gfx_target,
        "device_count": summary.device_count,
        "torch_version": summary.torch_version,
        "hip_version": summary.hip_version,
        "amdsmi_version": re.sub(r"\s+", " ", amdsmi).strip()[:500],
        "hipcc_version": [
            line.strip()
            for line in hipcc_output.splitlines()
            if "HIP version" in line
        ][:1],
    }


def _clock_state() -> dict[str, Any]:
    """Read current GFX clock + deep-sleep state (read-only, no sudo)."""
    state: dict[str, Any] = {"clock_locked_verified": False}
    try:
        metric = _run(
            [_required_rocm_tool("amd-smi"), "metric", "--clock"],
        ).stdout
        clk_match = re.search(r"CLK:\s*(\d+)\s*MHz", metric)
        sleep_match = re.search(r"DEEP_SLEEP:\s*(\w+)", metric)
        max_match = re.search(r"MAX_CLK:\s*(\d+)\s*MHz", metric)
        if clk_match:
            state["gfx_clock_mhz"] = int(clk_match.group(1))
        if max_match:
            state["gfx_max_clock_mhz"] = int(max_match.group(1))
        if sleep_match:
            state["deep_sleep"] = sleep_match.group(1)
            # Clocks count as locked/verified when not in deep sleep and running
            # near the forced peak (STABLE_PEAK disables deep sleep).
            state["clock_locked_verified"] = (
                state["deep_sleep"] == "DISABLED"
                and state.get("gfx_clock_mhz", 0) > 1000
            )
    except Exception as exc:  # noqa: BLE001 -- record diagnostic probe failure
        state["error"] = str(exc)
    return state


def _compile_probe(
    source: Path,
    workdir: Path,
    hipcc: str,
    architecture: str,
    *,
    compiler_defines: Mapping[str, int] | None = None,
) -> Path:
    definition_items = sorted((compiler_defines or {}).items())
    label = "-".join(
        f"{name.lower()}-{value}" for name, value in definition_items
    )
    binary_directory = workdir / "binaries" / source.stem
    binary_directory.mkdir(parents=True, exist_ok=True)
    binary = binary_directory / f"{source.stem}-{label or 'default'}.bin"
    _run(
        [
            hipcc,
            f"--offload-arch={architecture}",
            "-O3",
            *(f"-D{name}={value}" for name, value in definition_items),
            "-o",
            str(binary),
            str(source),
        ],
    )
    return binary


def _telemetry_snapshot(amdsmi: str) -> dict[str, Any]:
    """Capture a bounded, machine-readable environmental snapshot."""
    result = _run(
        [
            amdsmi,
            "metric",
            "--clock",
            "--temperature",
            "--power",
            "--perf-level",
            "--json",
            "--gpu",
            "0",
        ],
    )
    try:
        payload = json.loads(result.stdout)
        gpu = payload["gpu_data"][0]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("amd-smi returned invalid telemetry JSON") from exc
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "gfx_clock_mhz": _metric_value(gpu, "clock", "gfx_0", "clk"),
        "gfx_max_clock_mhz": _metric_value(gpu, "clock", "gfx_0", "max_clk"),
        "memory_clock_mhz": _metric_value(gpu, "clock", "mem_0", "clk"),
        "edge_temperature_c": _metric_value(gpu, "temperature", "edge"),
        "hotspot_temperature_c": _metric_value(gpu, "temperature", "hotspot"),
        "memory_temperature_c": _metric_value(gpu, "temperature", "mem"),
        "socket_power_w": _metric_value(gpu, "power", "socket_power"),
        "deep_sleep": _nested_value(gpu, "clock", "gfx_0", "deep_sleep"),
        "throttle_status": _nested_value(gpu, "power", "throttle_status"),
        "performance_level": gpu.get("perf_level"),
    }


def _nested_value(value: object, *keys: str) -> object:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _metric_value(value: object, *keys: str) -> float | None:
    metric = _nested_value(value, *keys)
    if isinstance(metric, Mapping):
        metric = metric.get("value")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
        return None
    return float(metric)


def _parse_result_samples(binary: Path, stdout: str) -> tuple[float, ...]:
    samples: list[float] = []
    for line in stdout.splitlines():
        if not line.startswith("RESULT "):
            continue
        try:
            samples.append(float(line.split("RESULT ", 1)[1]))
        except ValueError:
            continue
    if len(samples) != SAMPLES_PER_PROCESS_BATCH:
        raise RuntimeError(
            f"probe {binary.name} produced {len(samples)} RESULT lines; "
            f"expected {SAMPLES_PER_PROCESS_BATCH}",
        )
    if any(not np.isfinite(sample) or sample <= 0 for sample in samples):
        raise RuntimeError(
            f"probe {binary.name} produced invalid RESULT values",
        )
    return tuple(samples)


def _run_sample_batch(
    binary: Path,
    process_batch: int,
    *,
    amdsmi: str | None,
) -> SampleBatch:
    before = _telemetry_snapshot(amdsmi) if amdsmi is not None else None
    proc = _run([str(binary)])
    after = _telemetry_snapshot(amdsmi) if amdsmi is not None else None
    return SampleBatch(
        process_batch=process_batch,
        samples=_parse_result_samples(binary, proc.stdout),
        telemetry_before=before,
        telemetry_after=after,
    )


def _flatten_samples(batches: tuple[SampleBatch, ...]) -> tuple[float, ...]:
    samples = tuple(sample for batch in batches for sample in batch.samples)
    if not samples:
        raise RuntimeError("sample batches contain no RESULT values")
    return samples


def _bootstrap_median_interval(
    batch_medians: tuple[float, ...],
) -> dict[str, Any]:
    values = np.asarray(batch_medians, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(
        0,
        len(values),
        size=(BOOTSTRAP_REPLICATES, len(values)),
    )
    bootstrap_medians = np.median(values[indices], axis=1)
    lower, upper = np.percentile(bootstrap_medians, [2.5, 97.5])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": 0.95,
        "method": "percentile bootstrap over process-batch medians",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def _sample_statistics(batches: tuple[SampleBatch, ...]) -> dict[str, Any]:
    samples = _flatten_samples(batches)
    batch_medians = tuple(statistics.median(batch.samples) for batch in batches)
    lower_quartile, upper_quartile = np.percentile(samples, [25.0, 75.0])
    mean = statistics.mean(samples)
    standard_deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "primary_statistic": "median_of_process_batch_medians",
        "primary_result": statistics.median(batch_medians),
        "all_sample_median": statistics.median(samples),
        "all_sample_mean": mean,
        "all_sample_standard_deviation": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean,
        "minimum": min(samples),
        "maximum": max(samples),
        "lower_quartile": float(lower_quartile),
        "upper_quartile": float(upper_quartile),
        "interquartile_range": float(upper_quartile - lower_quartile),
        "batch_medians": list(batch_medians),
        "bootstrap_median_confidence_interval_95": _bootstrap_median_interval(
            batch_medians,
        ),
    }


def _telemetry_summary(batches: tuple[SampleBatch, ...]) -> dict[str, Any]:
    snapshots = [
        snapshot
        for batch in batches
        for snapshot in (batch.telemetry_before, batch.telemetry_after)
        if snapshot is not None
    ]
    numeric_fields = (
        "gfx_clock_mhz",
        "gfx_max_clock_mhz",
        "memory_clock_mhz",
        "edge_temperature_c",
        "hotspot_temperature_c",
        "memory_temperature_c",
        "socket_power_w",
    )
    numeric = {
        field: _numeric_summary(
            tuple(
                float(snapshot[field])
                for snapshot in snapshots
                if isinstance(snapshot.get(field), (int, float))
            ),
        )
        for field in numeric_fields
    }
    return {
        "snapshot_count": len(snapshots),
        "sampling": "immediately before and after each held-out process batch",
        "numeric": numeric,
        "deep_sleep_states": sorted(
            {
                str(item["deep_sleep"])
                for item in snapshots
                if item.get("deep_sleep")
            },
        ),
        "throttle_statuses": sorted(
            {
                str(item["throttle_status"])
                for item in snapshots
                if item.get("throttle_status")
            },
        ),
        "performance_levels": sorted(
            {
                str(item["performance_level"])
                for item in snapshots
                if item.get("performance_level")
            },
        ),
    }


def _numeric_summary(values: tuple[float, ...]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


# Instructions that distinguish every non-exempt native compute path on RDNA4.
CALIBRATION_ISA_CHECKS: tuple[ISAInstructionRequirement, ...] = (
    ISAInstructionRequirement("V_FMA_F32", "-"),
    ISAInstructionRequirement("V_FMAAK_F32", "-"),
    ISAInstructionRequirement("V_DUAL_FMAC_F32", "-"),
    ISAInstructionRequirement("V_PK_FMA_F16", "-"),
    ISAInstructionRequirement("V_PK_FMAC_F16", "-"),
    ISAInstructionRequirement("V_PK_FMA_BF16", "-"),
    ISAInstructionRequirement("V_PK_FMAC_BF16", "-"),
    ISAInstructionRequirement("V_PK_FMA_F32", "-"),
    ISAInstructionRequirement("V_WMMA_F32_16X16X16_F16", "WMMA"),
    ISAInstructionRequirement("V_WMMA_F32_16X16X16_BF16", "WMMA"),
    ISAInstructionRequirement("V_WMMA_F32_16X16X16_FP8_FP8", "WMMA"),
    ISAInstructionRequirement("V_WMMA_I32_16X16X16_IU8", "WMMA"),
)
CALIBRATION_INSTRUCTION_NAMES: tuple[str, ...] = tuple(
    requirement.instruction for requirement in CALIBRATION_ISA_CHECKS
)


def _isa_spec_evidence(architecture: str) -> dict[str, Any]:
    """Validate candidate instructions against the machine-readable ISA."""
    report = inspect_isa_requirements(architecture, CALIBRATION_ISA_CHECKS)
    supported = set(report.supported_instructions)
    return {
        **report.to_dict(),
        "instruction_presence": {
            name: name in supported for name in CALIBRATION_INSTRUCTION_NAMES
        },
    }


VALU_FP32_FALLBACK_INSTRUCTIONS = ("V_FMA_F32", "V_FMAAK_F32")
VALU_FP32_VOPD_INSTRUCTIONS = ("V_DUAL_FMAC_F32",)
VALU_FP16_INSTRUCTIONS = ("V_PK_FMA_F16", "V_PK_FMAC_F16")
VALU_BF16_INSTRUCTIONS = ("V_PK_FMA_BF16", "V_PK_FMAC_BF16")
WMMA_FP16_INSTRUCTIONS = ("V_WMMA_F32_16X16X16_F16",)
WMMA_BF16_INSTRUCTIONS = ("V_WMMA_F32_16X16X16_BF16",)
WMMA_FP8_INSTRUCTIONS = ("V_WMMA_F32_16X16X16_FP8_FP8",)
WMMA_INT8_INSTRUCTIONS = ("V_WMMA_I32_16X16X16_IU8",)


def _native_instruction_check(
    *,
    measurement: dict[str, Any],
    presence: dict[str, bool],
    instructions: tuple[str, ...],
) -> dict[str, Any]:
    declared = any(presence[name] for name in instructions)
    emitted = _count_emitted(measurement, instructions)
    runtime_passed = measurement["runtime_probe_passed"] is True
    passed = bool(declared and emitted > 0 and runtime_passed)
    return {
        "expectation": "native",
        "status": "passed" if passed else "failed",
        "probe": measurement["probe"],
        "instructions": list(instructions),
        "isa_declared": declared,
        "compiler_emitted_count": emitted,
        "runtime_probe_passed": runtime_passed,
        "native_instruction_usable": bool(declared and emitted > 0),
    }


def _bf16_fallback_check(
    *,
    measurement: dict[str, Any],
    presence: dict[str, bool],
) -> dict[str, Any]:
    declared = any(presence[name] for name in VALU_BF16_INSTRUCTIONS)
    emitted = _count_emitted(measurement, VALU_BF16_INSTRUCTIONS)
    fallback = _count_emitted(measurement, VALU_FP32_FALLBACK_INSTRUCTIONS)
    runtime_passed = measurement["runtime_probe_passed"] is True
    passed = bool(
        not declared and emitted == 0 and fallback > 0 and runtime_passed,
    )
    return {
        "expectation": "fallback",
        "status": "passed" if passed else "failed",
        "probe": measurement["probe"],
        "instructions": list(VALU_BF16_INSTRUCTIONS),
        "fallback_instructions": list(VALU_FP32_FALLBACK_INSTRUCTIONS),
        "isa_declared": declared,
        "compiler_emitted_count": emitted,
        "fallback_emitted_count": fallback,
        "runtime_probe_passed": runtime_passed,
        "native_instruction_usable": bool(declared and emitted > 0),
    }


def _instruction_check_specs() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return (
        ("fp32_valu_fma", "vector_fp32_fp32.hip", VALU_FP32_VOPD_INSTRUCTIONS),
        (
            "fp16_packed_valu_fma",
            "vector_fp16_fp16.hip",
            VALU_FP16_INSTRUCTIONS,
        ),
        ("fp16_wmma", "matrix_fp16_fp16_wmma.hip", WMMA_FP16_INSTRUCTIONS),
        ("bf16_wmma", "matrix_bf16_bf16_wmma.hip", WMMA_BF16_INSTRUCTIONS),
        ("fp8_wmma", "matrix_fp8_fp8_wmma.hip", WMMA_FP8_INSTRUCTIONS),
        ("int8_wmma", "matrix_int8_int8_wmma.hip", WMMA_INT8_INSTRUCTIONS),
    )


def _instruction_validation(
    spec_evidence: dict[str, Any],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cross-check ISA declaration, compiler emission, and runtime execution."""
    presence = spec_evidence["instruction_presence"]
    by_probe = {
        measurement["probe"]: measurement for measurement in measurements
    }
    checks = {
        name: _native_instruction_check(
            measurement=by_probe[probe],
            presence=presence,
            instructions=instructions,
        )
        for name, probe, instructions in _instruction_check_specs()
    }
    checks["bf16_valu_fallback"] = _bf16_fallback_check(
        measurement=by_probe["vector_bf16_bf16.hip"],
        presence=presence,
    )
    failed = sorted(
        name for name, check in checks.items() if check["status"] != "passed"
    )
    if failed or tuple(sorted(checks)) != REQUIRED_INSTRUCTION_CHECKS:
        raise RuntimeError(
            f"ISA/spec, compiler-emission, and runtime checks disagree: failed={failed}",
        )
    return {
        "status": "passed",
        "required_checks": list(REQUIRED_INSTRUCTION_CHECKS),
        "checks": checks,
        "conclusion": (
            "gfx1200 declares, emits, and runs native FP32 VOPD, packed FP16 "
            "VALU, plus "
            "FP16/BF16/FP8/INT8 WMMA instructions; packed BF16 VALU is absent "
            "from the ISA and code object and executes through FP32 fallback"
        ),
    }


def _calibration_coverage(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    covered_precisions = sorted(
        {
            precision
            for measurement in measurements
            for precision in measurement["covers_precisions"]
        },
    )
    covered_resources = sorted(
        {
            target
            for measurement in measurements
            for target in measurement["covers_resource_modes"]
        },
    )
    if (
        tuple(covered_precisions) != REQUIRED_PRECISIONS
        or tuple(covered_resources) != REQUIRED_RESOURCE_MODES
    ):
        raise RuntimeError(
            "calibration probes do not exactly cover the declared requirements",
        )
    return {
        "status": "passed",
        "required_precisions": list(REQUIRED_PRECISIONS),
        "covered_precisions": covered_precisions,
        "required_resource_modes": list(REQUIRED_RESOURCE_MODES),
        "covered_resource_modes": covered_resources,
    }


def _compiler_isa_evidence(
    binary: Path,
    architecture: str,
    workspace: Path,
    tracked_instructions: tuple[str, ...],
) -> dict[str, Any]:
    """Extract emitted GPU ISA and count the instructions relevant to a probe."""
    extracted = extract_code_object(
        binary,
        architecture,
        workspace,
        timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    )
    analysis = analyze_isa_disassembly(
        architecture,
        extracted.disassembly,
        expected_instructions=tracked_instructions,
    )
    return {
        "architecture": extracted.architecture,
        "code_object_sha256": extracted.sha256,
        "disassembly_sha256": extracted.disassembly_sha256,
        "decoded_instruction_count": analysis.decoded_instruction_count,
        "matched_instruction_counts": dict(analysis.matched_instruction_counts),
        "spec_provenance": analysis.provenance.to_dict(),
    }


def _count_emitted(measurement: dict[str, Any], names: tuple[str, ...]) -> int:
    counts = measurement["compiler_isa"]["matched_instruction_counts"]
    return sum(int(counts.get(name, 0)) for name in names)


def _tuning_candidate_entry(
    tuning: TuningSpec,
    candidate: int,
    binary: Path,
    batches: tuple[SampleBatch, ...],
) -> dict[str, Any]:
    samples = _flatten_samples(batches)
    batch_medians = tuple(statistics.median(batch.samples) for batch in batches)
    lower_quartile, upper_quartile = np.percentile(samples, [25.0, 75.0])
    return {
        "configuration": {tuning.parameter: candidate},
        "compiler_defines": {tuning.compiler_macro: candidate},
        "binary_sha256": sha256_file(binary),
        "process_batch_count": len(batches),
        "sample_count": len(samples),
        "batch_medians": list(batch_medians),
        "selection_result": statistics.median(batch_medians),
        "all_sample_median": statistics.median(samples),
        "interquartile_range": float(upper_quartile - lower_quartile),
        "raw_process_batches": [batch.to_dict() for batch in batches],
    }


def _tune_probe(
    probe: Mapping[str, Any],
    source: Path,
    workdir: Path,
    hipcc: str,
    architecture: str,
    tuning_batches: int,
) -> PreparedProbe:
    tuning = cast(TuningSpec, probe["tuning"])
    binaries = {
        candidate: _compile_probe(
            source,
            workdir,
            hipcc,
            architecture,
            compiler_defines={tuning.compiler_macro: candidate},
        )
        for candidate in tuning.candidates
    }
    observed: dict[int, list[SampleBatch]] = {
        candidate: [] for candidate in tuning.candidates
    }
    execution_order: list[dict[str, int]] = []
    for tuning_round in range(tuning_batches):
        order = (
            tuning.candidates
            if tuning_round % 2 == 0
            else tuple(reversed(tuning.candidates))
        )
        for candidate in order:
            observed[candidate].append(
                _run_sample_batch(
                    binaries[candidate],
                    tuning_round,
                    amdsmi=None,
                ),
            )
            execution_order.append(
                {"tuning_round": tuning_round, tuning.parameter: candidate},
            )
    candidates = [
        _tuning_candidate_entry(
            tuning,
            candidate,
            binaries[candidate],
            tuple(observed[candidate]),
        )
        for candidate in tuning.candidates
    ]
    selected = max(
        candidates,
        key=lambda item: (
            float(item["selection_result"]),
            -int(item["configuration"][tuning.parameter]),
        ),
    )
    selected_value = int(selected["configuration"][tuning.parameter])
    return PreparedProbe(
        specification=probe,
        source=source,
        binary=binaries[selected_value],
        compiler_defines={tuning.compiler_macro: selected_value},
        selected_configuration={tuning.parameter: selected_value},
        tuning_evidence={
            "status": "performed",
            "phase": "configuration_selection_only",
            "search_method": "deterministic exhaustive search",
            "parameter": tuning.parameter,
            "candidate_values": list(tuning.candidates),
            "selection_rule": (
                "highest median of process-batch medians; smallest candidate "
                "wins an exact tie"
            ),
            "counterbalancing": "ascending and descending candidate order alternates",
            "execution_order": execution_order,
            "candidates": candidates,
            "selected_configuration": {tuning.parameter: selected_value},
            "held_out_samples_used_for_selection": False,
        },
    )


def _prepare_probe(
    probe: Mapping[str, Any],
    workdir: Path,
    hipcc: str,
    architecture: str,
    tuning_batches: int,
) -> PreparedProbe:
    source = PROBE_DIR / str(probe["source"])
    if isinstance(probe.get("tuning"), TuningSpec):
        prepared = _tune_probe(
            probe,
            source,
            workdir,
            hipcc,
            architecture,
            tuning_batches,
        )
        selected = prepared.selected_configuration
        print(
            f"[tune] {source.name}: selected {dict(selected)}",
            file=sys.stderr,
        )
        return prepared
    binary = _compile_probe(source, workdir, hipcc, architecture)
    return PreparedProbe(
        specification=probe,
        source=source,
        binary=binary,
        compiler_defines={},
        selected_configuration={},
        tuning_evidence={
            "status": "not_required",
            "phase": "configuration_selection_only",
            "search_method": None,
            "selected_configuration": {},
            "held_out_samples_used_for_selection": False,
        },
    )


def _measure_held_out(
    probes: tuple[PreparedProbe, ...],
    process_batches: int,
    amdsmi: str,
) -> tuple[dict[str, tuple[SampleBatch, ...]], list[dict[str, Any]]]:
    collected: dict[str, list[SampleBatch]] = {
        str(probe.specification["source"]): [] for probe in probes
    }
    execution_order: list[dict[str, Any]] = []
    for process_batch in range(process_batches):
        offset = process_batch % len(probes)
        order = probes[offset:] + probes[:offset]
        if process_batch % 2 == 1:
            order = tuple(reversed(order))
        for position, probe in enumerate(order):
            source_name = str(probe.specification["source"])
            collected[source_name].append(
                _run_sample_batch(
                    probe.binary,
                    process_batch,
                    amdsmi=amdsmi,
                ),
            )
            execution_order.append(
                {
                    "process_batch": process_batch,
                    "position": position,
                    "probe": source_name,
                },
            )
    return (
        {name: tuple(batches) for name, batches in collected.items()},
        execution_order,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="RX_9060_XT")
    parser.add_argument(
        "--gfx",
        default="gfx1200",
        help="ISA target / offload arch",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "src"
        / "solar"
        / "audits"
        / "rx9060xt_resource_peaks_v3.json",
    )
    parser.add_argument(
        "--tuning-batches",
        type=int,
        default=3,
        help="fresh processes per tuning candidate",
    )
    parser.add_argument(
        "--measurement-batches",
        "--repeats",
        dest="measurement_batches",
        type=int,
        default=7,
        help="fresh held-out processes per frozen probe (minimum: 5)",
    )
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument(
        "--allow-unlocked",
        action="store_true",
        help="do not fail when clocks look unlocked",
    )
    args = parser.parse_args(argv)
    if args.tuning_batches <= 0:
        parser.error("--tuning-batches must be positive")
    if args.measurement_batches < MIN_HELD_OUT_PROCESS_BATCHES:
        parser.error(
            f"--measurement-batches must be at least {MIN_HELD_OUT_PROCESS_BATCHES}",
        )
    return args


def _probe_entry(
    prepared: PreparedProbe,
    *,
    batches: tuple[SampleBatch, ...],
    architecture: str,
    workdir: Path,
) -> dict[str, Any]:
    probe = prepared.specification
    source = prepared.source
    binary = prepared.binary
    nominal = probe["nominal_ops_per_second"]
    sample_statistics = _sample_statistics(batches)
    primary_result = float(sample_statistics["primary_result"])
    best_observed_result = float(sample_statistics["maximum"])
    result_scale = float(probe["ops_per_result"])
    ops_per_second = primary_result * result_scale
    best_observed_ops_per_second = best_observed_result * result_scale
    sample_count = sum(len(batch.samples) for batch in batches)
    entry = {
        "probe": probe["source"],
        "source_sha256": sha256_file(source),
        "binary_sha256": sha256_file(binary),
        "compiler_defines": dict(prepared.compiler_defines),
        "selected_configuration": dict(prepared.selected_configuration),
        "tuning": dict(prepared.tuning_evidence),
        "resource": probe["resource"],
        "mode": probe["mode"],
        "covers_precisions": list(probe["covers_precisions"]),
        "covers_resource_modes": list(probe["covers_resource_modes"]),
        "result_unit": (
            "ops/s" if probe["resource"] != "bandwidth" else "bytes/s"
        ),
        "result_scale": result_scale,
        "measurement_phase": "held_out_after_configuration_freeze",
        "primary_statistic": sample_statistics["primary_statistic"],
        "peak_result": best_observed_result,
        "median_result": primary_result,
        "minimum_result": sample_statistics["minimum"],
        "sample_count": sample_count,
        "process_batch_count": len(batches),
        "samples_per_process_batch": SAMPLES_PER_PROCESS_BATCH,
        "raw_process_batches": [batch.to_dict() for batch in batches],
        "statistics": sample_statistics,
        "telemetry_summary": _telemetry_summary(batches),
        "runtime_probe_passed": True,
        "measured_ops_per_second": ops_per_second,
        "best_observed_ops_per_second": best_observed_ops_per_second,
        "nominal_ops_per_second": nominal,
        "nominal_source": probe["nominal_source"],
        "measured_to_nominal_ratio": (
            ops_per_second / nominal if nominal else None
        ),
        "best_observed_to_nominal_ratio": (
            best_observed_ops_per_second / nominal if nominal else None
        ),
    }
    tracked = probe.get("tracked_instructions")
    if tracked:
        entry["compiler_isa"] = _compiler_isa_evidence(
            binary,
            architecture,
            workdir / "isa" / source.stem,
            tuple(tracked),
        )
    return entry


def _collect_measurements(
    args: argparse.Namespace,
    workdir: Path,
    hipcc: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared = tuple(
        _prepare_probe(
            probe,
            workdir,
            hipcc,
            args.gfx,
            args.tuning_batches,
        )
        for probe in PROBES
    )
    held_out, execution_order = _measure_held_out(
        prepared,
        args.measurement_batches,
        _required_rocm_tool("amd-smi"),
    )
    measurements: list[dict[str, Any]] = []
    contradictions: list[str] = []
    print(
        f"{'probe':28s} {'resource':10s} {'mode':5s} "
        f"{'primary':>14s} {'best':>14s} {'nominal':>14s} {'ratio':>7s}",
        file=sys.stderr,
    )
    for prepared_probe in prepared:
        probe = prepared_probe.specification
        source_name = str(probe["source"])
        entry = _probe_entry(
            prepared_probe,
            batches=held_out[source_name],
            architecture=args.gfx,
            workdir=workdir,
        )
        measurements.append(entry)
        nominal = entry["nominal_ops_per_second"]
        measured = entry["measured_ops_per_second"]
        nominal_str = f"{nominal:.4e}" if nominal else "n/a"
        ratio_str = f"{measured / nominal:.2f}" if nominal else "n/a"
        print(
            f"{probe['source']:28s} {probe['resource']:10s} {probe['mode']:5s} "
            f"{measured:14.4e} {entry['best_observed_ops_per_second']:14.4e} "
            f"{nominal_str:>14s} {ratio_str:>7s}",
            file=sys.stderr,
        )
        best_observed = float(entry["best_observed_ops_per_second"])
        if nominal is not None and best_observed > nominal * 1.001:
            contradictions.append(
                f"{probe['resource']}/{probe['mode']} measured "
                f"{best_observed:.4e} > "
                f"nominal {nominal:.4e} ({probe['nominal_source']})",
            )
    if contradictions:
        details = "\n".join(f"  - {item}" for item in contradictions)
        raise RuntimeError(
            "calibration contradicts one or more nominal limits:\n" + details,
        )
    return measurements, execution_order


def _build_artifact(
    args: argparse.Namespace,
    *,
    clock: dict[str, Any],
    device: dict[str, Any],
    hipcc: str,
    measurements: list[dict[str, Any]],
    spec_evidence: dict[str, Any],
    held_out_execution_order: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "timing_profile": TIMING_PROFILE,
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "device": device,
        "clock_setup": {
            "clock_locked_verified": clock.get("clock_locked_verified", False),
            "gfx_clock_mhz": clock.get("gfx_clock_mhz"),
            "gfx_max_clock_mhz": clock.get("gfx_max_clock_mhz"),
            "deep_sleep": clock.get("deep_sleep"),
            "method": (
                "amd-smi metric --clock (STABLE_PEAK perf level disables deep sleep)"
            ),
        },
        "toolchain": {
            "hipcc": hipcc,
            "offload_arch": args.gfx,
            "optimization": "-O3",
            "numpy_version": np.__version__,
        },
        "experiment_protocol": {
            "design": "two_phase_tuning_then_held_out_measurement",
            "tuning_search": "deterministic exhaustive search",
            "tuning_process_batches_per_candidate": args.tuning_batches,
            "held_out_process_batches_per_probe": args.measurement_batches,
            "samples_per_process_batch": SAMPLES_PER_PROCESS_BATCH,
            "candidate_order": "ascending/descending counterbalanced by tuning round",
            "held_out_probe_order": (
                "rotated and direction-alternated by process batch"
            ),
            "held_out_execution_order": held_out_execution_order,
            "primary_statistic": "median_of_process_batch_medians",
            "uncertainty": (
                "95% deterministic percentile bootstrap interval over "
                "process-batch medians"
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "raw_samples_retained": True,
            "configuration_frozen_before_held_out_measurement": True,
            "telemetry_sampling": (
                "amd-smi JSON immediately before and after every held-out probe process"
            ),
        },
        "source_revision": _git_revision(),
        "calibration_script_sha256": sha256_file(Path(__file__)),
        "measurements": measurements,
        "calibration_coverage": _calibration_coverage(measurements),
        "isa_spec_evidence": spec_evidence,
        "instruction_validation": _instruction_validation(
            spec_evidence,
            measurements,
        ),
        "policy": (
            "SOL bounds use the AMD nominal theoretical peak; the sustained "
            "held-out median here is audit evidence only and must not replace "
            "the nominal bound. Best-observed throughput is retained as "
            "supplemental evidence, never as the primary statistic."
        ),
    }


def _write_artifact(args: argparse.Namespace, artifact: dict[str, Any]) -> None:
    artifact["payload_sha256"] = resource_peak_payload_sha256(artifact)
    atomic_write_json_value(args.out, artifact)
    digest = sha256_file(args.out)
    print(f"\n[calibrate] wrote {args.out}", file=sys.stderr)
    print(f"[calibrate] file sha256 = {digest}", file=sys.stderr)
    try:
        relative_path = args.out.resolve().relative_to(SOLAR_ROOT.resolve())
    except ValueError:
        print(
            "[calibrate] output is outside src/solar; no profile patch emitted",
            file=sys.stderr,
        )
        return
    throttle_statuses = {
        str(snapshot["throttle_status"])
        for measurement in artifact["measurements"]
        for batch in measurement["raw_process_batches"]
        for snapshot in (batch["telemetry_before"], batch["telemetry_after"])
    }
    required_unthrottled = throttle_statuses == {"UNTHROTTLED"}
    evidence_scope = (
        "unthrottled_resource_peak"
        if required_unthrottled
        else "instruction_and_runtime_corroboration_only"
    )
    print(
        "\n# ---- patch for src/solar/rocm/profiles/RX_9060_XT.yaml ----",
        file=sys.stderr,
    )
    print(
        "audit_evidence:\n"
        "  status: verified\n"
        f"  path: {relative_path}\n"
        f"  sha256: {digest}\n"
        f"  required_schema_version: {SCHEMA_VERSION}\n"
        f"  required_timing_profile: {TIMING_PROFILE}\n"
        "  required_clocks_locked: true\n"
        f"  required_unthrottled: {str(required_unthrottled).lower()}\n"
        f"  evidence_scope: {evidence_scope}\n"
        f"  gfx_target: {args.gfx}",
        file=sys.stderr,
    )


def _calibrate(args: argparse.Namespace, workdir: Path) -> int:
    hipcc = _required_rocm_tool("hipcc")
    workdir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("[calibrate] verifying clock state...", file=sys.stderr)
    clock = _clock_state()
    if not clock.get("clock_locked_verified") and not args.allow_unlocked:
        print(
            f"[calibrate] ERROR: clocks not verified locked ({clock}); "
            "run `sudo /opt/rocm/bin/amd-smi set -l STABLE_PEAK` first or pass --allow-unlocked.",
            file=sys.stderr,
        )
        return 2

    device = _device_identity()
    if device["gfx_target"] != args.gfx:
        raise RuntimeError(
            f"visible GPU is {device['gfx_target']}, expected {args.gfx}",
        )
    spec_evidence = _isa_spec_evidence(args.gfx)
    measurements, held_out_execution_order = _collect_measurements(
        args,
        workdir,
        hipcc,
    )
    artifact = _build_artifact(
        args,
        clock=clock,
        device=device,
        hipcc=hipcc,
        measurements=measurements,
        spec_evidence=spec_evidence,
        held_out_execution_order=held_out_execution_order,
    )
    _write_artifact(args, artifact)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run resource-peak calibration and persist audited evidence."""
    args = _parse_args(argv)
    if args.workdir is not None:
        return _calibrate(args, args.workdir)
    with tempfile.TemporaryDirectory(
        prefix="solar-resource-peak-calibration-",
    ) as temporary:
        return _calibrate(args, Path(temporary))


if __name__ == "__main__":
    raise SystemExit(main())
