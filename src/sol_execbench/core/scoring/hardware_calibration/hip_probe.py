# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Small, injectable HIP calibration probe abstraction.

The production implementation deliberately has no import-time HIP dependency.
Callers provide compilation and execution seams, which keeps calibration tests
portable and prevents a missing ROCm installation from becoming fabricated data.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import replace
from importlib import resources
from math import isfinite
from pathlib import Path
from dataclasses import dataclass
from tempfile import mkdtemp
from typing import Any, Callable, Mapping

from sol_execbench.core.platform.amdgpu_code_object import extract_code_object
from sol_execbench.core.platform.isa_validation import (
    IsaInstructionRequirement,
    analyze_isa_disassembly,
    inspect_isa_requirements,
)
from sol_execbench.core.scoring.hardware_calibration.models import (
    CalibrationCandidate,
    CalibrationIsaValidation,
)
from sol_execbench.tools.amd_isa import IsaError
from sol_execbench.core.platform.runtime import resolve_rocm_tool
from sol_execbench.core.scoring.hardware_calibration.statistics import (
    MINIMUM_SAMPLE_COUNT,
    select_conservative_value,
)


@dataclass(frozen=True, order=True)
class CalibrationProfileKey:
    """Identifies one measured calibration matrix entry."""

    kind: str
    operation: str
    input_dtype: str
    output_dtype: str
    path: str

    @property
    def value(self) -> str:
        return ".".join(
            (self.kind, self.operation, self.input_dtype, self.output_dtype, self.path)
        )


@dataclass(frozen=True)
class ProbeExecution:
    """Timed samples produced by a successfully executed probe."""

    samples: tuple[float, ...]
    unit: str
    correct: bool = True
    stable: bool = True


CompileCandidate = Callable[[CalibrationProfileKey], str | bool | None]
ExecuteCandidate = Callable[[CalibrationProfileKey], ProbeExecution | None]
CheckCandidate = Callable[[CalibrationProfileKey, ProbeExecution], bool | None]
ProbeEvidence = Callable[[CalibrationProfileKey], Mapping[str, Any] | None]
IsaPreflight = Callable[[CalibrationProfileKey, IsaInstructionRequirement], bool]
CompiledIsaValidator = Callable[
    [CalibrationProfileKey, Path, IsaInstructionRequirement], bool
]
_DEFAULT_HIPCC = object()
_PROBE_RESOURCE_PACKAGE = "sol_execbench.data.hardware_calibration_probes"
_PROBE_RESOURCES: dict[tuple[str, str, str, str | None], str] = {
    ("vector", "fp32", "fp32", None): "vector_fp32_fp32.hip",
    ("stream_copy", "fp32", "fp32", None): "stream_copy_fp32_fp32.hip",
    ("vector", "fp16", "fp16", None): "vector_fp16_fp16.hip",
    ("stream_copy", "fp16", "fp16", None): "stream_copy_fp16_fp16.hip",
    ("vector", "bf16", "bf16", None): "vector_bf16_bf16.hip",
    ("stream_copy", "bf16", "bf16", None): "stream_copy_bf16_bf16.hip",
    ("stream_copy", "bf16", "fp32", None): "stream_copy_bf16_fp32.hip",
    ("stream_copy", "fp32", "bf16", None): "stream_copy_fp32_bf16.hip",
    ("reduction", "fp32", "fp32", None): "reduction_fp32_fp32.hip",
    ("transcendental", "fp32", "fp32", None): "transcendental_fp32_fp32.hip",
    # gfx12 FP32 matrix math uses the ordinary FP32 FMA datapath rather than
    # WMMA. Reuse the self-checking saturation probe for that shared resource.
    ("matrix", "fp32", "fp32", "gfx12"): "vector_fp32_fp32.hip",
    ("matrix", "bf16", "bf16", "wmma"): "matrix_bf16_bf16_wmma.hip",
    ("matrix", "fp16", "fp16", "wmma"): "matrix_fp16_fp16_wmma.hip",
    ("matrix", "bf16", "bf16", "mfma"): "matrix_bf16_bf16_mfma.hip",
}

_MATRIX_ISA_REQUIREMENTS = {
    "compute.matrix.bf16.bf16.wmma": IsaInstructionRequirement(
        "V_WMMA_F32_16X16X16_BF16", "WMMA"
    ),
    "compute.matrix.fp16.fp16.wmma": IsaInstructionRequirement(
        "V_WMMA_F32_16X16X16_F16", "WMMA"
    ),
    "compute.matrix.bf16.bf16.mfma": IsaInstructionRequirement(
        "V_MFMA_F32_16X16X16_BF16", "MFMA"
    ),
}


def matrix_isa_requirements() -> tuple[IsaInstructionRequirement, ...]:
    """Return exact ISA requirements declared by packaged matrix probes."""

    return tuple(_MATRIX_ISA_REQUIREMENTS.values())


@dataclass(frozen=True)
class HipProbe:
    """Compile, run, check, and stabilise a HIP candidate through injected seams."""

    compile_candidate: CompileCandidate | None = None
    execute_candidate: ExecuteCandidate | None = None
    check_correctness: CheckCandidate | None = None
    check_stability: CheckCandidate | None = None
    evidence_for: ProbeEvidence | None = None

    def provenance_for(self, key: CalibrationProfileKey) -> dict[str, Any]:
        """Return compile/run provenance when the concrete backend exposes it."""
        if self.evidence_for is None:
            return {"collection": "injected_probe"}
        try:
            evidence = self.evidence_for(key)
        except Exception:
            return {"collection": "unavailable"}
        return (
            dict(evidence)
            if isinstance(evidence, Mapping)
            else {"collection": "unavailable"}
        )

    def measure(self, key: CalibrationProfileKey) -> CalibrationCandidate:
        compile_state = self._compile(key)
        if compile_state == "unsupported":
            return self._with_isa_validation(
                self._unavailable(key, "hip_probe_explicitly_unsupported"), key
            )
        if compile_state != "passed":
            return self._with_isa_validation(
                self._unknown(key, f"hip_probe_compile_{compile_state}"), key
            )
        execution = self._execute(key)
        if execution is None:
            return self._with_isa_validation(
                self._unknown(key, "hip_probe_run_failed"), key
            )
        if not isinstance(execution, ProbeExecution):
            return self._with_isa_validation(
                self._unknown(key, "hip_probe_execution_malformed"), key
            )
        if not self._check(self.check_correctness, key, execution):
            return self._with_isa_validation(
                self._unknown(key, "hip_probe_correctness_failed"), key
            )
        if not self._check(self.check_stability, key, execution):
            return self._with_isa_validation(
                self._unknown(key, "hip_probe_stability_failed"), key
            )
        try:
            return self._with_isa_validation(
                CalibrationCandidate(
                    key=key.value,
                    state="measured",
                    value=select_conservative_value(execution.samples).value,
                    unit=execution.unit,
                    samples=execution.samples,
                ),
                key,
            )
        except (TypeError, ValueError):
            return self._with_isa_validation(
                self._unknown(key, "hip_probe_samples_invalid"), key
            )

    def _with_isa_validation(
        self, candidate: CalibrationCandidate, key: CalibrationProfileKey
    ) -> CalibrationCandidate:
        evidence = self.provenance_for(key)
        raw = evidence.get("isa_validation")
        if not isinstance(raw, Mapping):
            return candidate
        try:
            validation = CalibrationIsaValidation(
                status=str(raw["status"]),
                architecture=str(raw["architecture"]),
                expected_instruction=_optional_text(raw.get("expected_instruction")),
                expected_subgroup=_optional_text(raw.get("expected_subgroup")),
                matched_instruction_count=int(raw.get("matched_instruction_count", 0)),
                code_object_path=_optional_text(raw.get("code_object_path")),
                code_object_sha256=_optional_text(raw.get("code_object_sha256")),
                disassembly_path=_optional_text(raw.get("disassembly_path")),
                disassembly_sha256=_optional_text(raw.get("disassembly_sha256")),
                spec_provenance={
                    str(item_key): str(item)
                    for item_key, item in dict(raw.get("spec_provenance", {})).items()
                },
                reason_code=_optional_text(raw.get("reason_code")),
            )
        except (KeyError, TypeError, ValueError):
            return candidate
        return replace(candidate, isa_validation=validation)

    def _compile(self, key: CalibrationProfileKey) -> str:
        if self.compile_candidate is None:
            return "missing"
        try:
            result = self.compile_candidate(key)
        except Exception:
            return "failed"
        if result in {"unsupported", "missing", "failed"}:
            return result
        return "passed" if result is True or result == "passed" else "failed"

    def _execute(self, key: CalibrationProfileKey) -> ProbeExecution | None:
        if self.execute_candidate is None:
            return None
        try:
            return self.execute_candidate(key)
        except Exception:
            return None

    @staticmethod
    def _check(
        check: CheckCandidate | None,
        key: CalibrationProfileKey,
        execution: ProbeExecution,
    ) -> bool:
        if check is None:
            return False
        try:
            return check(key, execution) is True
        except Exception:
            return False

    @staticmethod
    def _unknown(key: CalibrationProfileKey, reason_code: str) -> CalibrationCandidate:
        return CalibrationCandidate(
            key.value, "unknown", None, None, reason_code=reason_code
        )

    @staticmethod
    def _unavailable(
        key: CalibrationProfileKey, reason_code: str
    ) -> CalibrationCandidate:
        return CalibrationCandidate(
            key.value, "unavailable", None, None, reason_code=reason_code
        )


class HipCommandBackend:
    """Concrete HIPCC backend, lazy enough to remain usable off ROCm hosts."""

    def __init__(
        self,
        workspace: Path | None = None,
        hipcc: str | None | object = _DEFAULT_HIPCC,
        architecture: str | None = None,
        run: Callable[..., object] = subprocess.run,
        allow_isa_download: bool = True,
        isa_preflight: IsaPreflight | None = None,
        compiled_isa_validator: CompiledIsaValidator | None = None,
    ) -> None:
        self.workspace = workspace or Path(mkdtemp(prefix="sol-execbench-hip-"))
        if hipcc is _DEFAULT_HIPCC:
            discovered_hipcc = resolve_rocm_tool("hipcc")
            self.hipcc = str(discovered_hipcc) if discovered_hipcc else None
        else:
            self.hipcc = hipcc
        self.architecture = architecture.lower() if architecture is not None else None
        self.run = run
        self.allow_isa_download = allow_isa_download
        self.isa_preflight = isa_preflight
        self.compiled_isa_validator = compiled_isa_validator
        self._executables: dict[CalibrationProfileKey, Path] = {}
        self._evidence: dict[CalibrationProfileKey, dict[str, Any]] = {}

    def compile(self, key: CalibrationProfileKey) -> str:
        if self.hipcc is None:
            return "missing"
        requirement = _MATRIX_ISA_REQUIREMENTS.get(key.value)
        if requirement is not None:
            supported = (self.isa_preflight or self._preflight_isa)(key, requirement)
            if not supported:
                return "unsupported"
        if not _is_supported_probe(key):
            return "failed"
        self.workspace.mkdir(parents=True, exist_ok=True)
        stem = key.value.replace(".", "_")
        source = self.workspace / f"{stem}.hip"
        executable = self.workspace / stem
        try:
            source_text = _hip_source(key)
            source.write_text(source_text, encoding="utf-8")
            command = [self.hipcc, "-O3"]
            if self.architecture is not None:
                command.append(f"--offload-arch={self.architecture}")
            command.extend((str(source), "-o", str(executable)))
            result = self.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "failed"
        if getattr(result, "returncode", 1) != 0:
            return "failed"
        self._executables[key] = executable
        try:
            binary_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
        except OSError:
            binary_sha256 = None
        self._evidence[key] = {
            "compiler_command": command,
            "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "binary_sha256": binary_sha256,
            "warmup_iterations": 3,
            "timed_samples": 7,
        }
        validator = self.compiled_isa_validator or self._validate_compiled_isa
        if requirement is not None and not validator(key, executable, requirement):
            return "failed"
        return "passed"

    def _preflight_isa(
        self, key: CalibrationProfileKey, requirement: IsaInstructionRequirement
    ) -> bool:
        assert self.architecture is not None
        try:
            report = inspect_isa_requirements(
                self.architecture,
                (requirement,),
                allow_download=self.allow_isa_download,
            )
        except IsaError as exc:
            self._set_isa_failure(key, requirement, "unavailable", _isa_error_code(exc))
            return False
        if not report.supports(requirement):
            self._set_isa_failure(
                key, requirement, "unavailable", "isa_instruction_unsupported"
            )
            return False
        return True

    def _validate_compiled_isa(
        self,
        key: CalibrationProfileKey,
        executable: Path,
        requirement: IsaInstructionRequirement,
    ) -> bool:
        assert self.architecture is not None
        try:
            extracted = extract_code_object(
                executable,
                self.architecture,
                self.workspace / "isa-artifacts" / key.value,
            )
            disassembly_path = extracted.path.with_suffix(".disassembly.txt")
            disassembly_path.write_text(extracted.disassembly, encoding="utf-8")
            analysis = analyze_isa_disassembly(
                self.architecture,
                extracted.disassembly,
                expected_instructions=(requirement.instruction,),
                allow_download=self.allow_isa_download,
            )
        except (IsaError, OSError, RuntimeError, ValueError) as exc:
            self._set_isa_failure(key, requirement, "failed", _isa_error_code(exc))
            return False
        matched = analysis.matched_instruction_counts.get(requirement.instruction, 0)
        self._evidence.setdefault(key, {})["isa_validation"] = {
            "status": "verified" if matched > 0 else "failed",
            "architecture": self.architecture,
            "expected_instruction": requirement.instruction,
            "expected_subgroup": requirement.functional_subgroup,
            "matched_instruction_count": matched,
            "code_object_path": str(extracted.path),
            "code_object_sha256": extracted.sha256,
            "disassembly_path": str(disassembly_path),
            "disassembly_sha256": extracted.disassembly_sha256,
            "spec_provenance": analysis.provenance.to_dict(),
            "reason_code": None if matched > 0 else "isa_expected_instruction_missing",
        }
        return matched > 0

    def _set_isa_failure(
        self,
        key: CalibrationProfileKey,
        requirement: IsaInstructionRequirement,
        status: str,
        reason_code: str,
    ) -> None:
        self._evidence.setdefault(key, {})["isa_validation"] = {
            "status": status,
            "architecture": self.architecture or "unknown",
            "expected_instruction": requirement.instruction,
            "expected_subgroup": requirement.functional_subgroup,
            "matched_instruction_count": 0,
            "code_object_path": None,
            "code_object_sha256": None,
            "disassembly_path": None,
            "disassembly_sha256": None,
            "spec_provenance": {},
            "reason_code": reason_code,
        }

    def evidence(self, key: CalibrationProfileKey) -> Mapping[str, Any] | None:
        return self._evidence.get(key)

    def execute(self, key: CalibrationProfileKey) -> ProbeExecution | None:
        executable = self._executables.get(key)
        if executable is None:
            return None
        try:
            result = self.run(
                [str(executable)], capture_output=True, text=True, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if getattr(result, "returncode", 1) != 0:
            return None
        try:
            samples = tuple(
                float(line.split()[1])
                for line in str(getattr(result, "stdout", "")).splitlines()
                if line.startswith("RESULT ")
            )
        except (IndexError, ValueError):
            return None
        if not samples or any(
            not isfinite(sample) or sample <= 0.0 for sample in samples
        ):
            return None
        return ProbeExecution(
            samples=samples,
            unit="TFLOP/s" if key.kind == "compute" else "GB/s",
        )

    @staticmethod
    def correctness(_: CalibrationProfileKey, execution: ProbeExecution) -> bool:
        return execution.correct

    @staticmethod
    def stability(_: CalibrationProfileKey, execution: ProbeExecution) -> bool:
        return execution.stable and len(execution.samples) >= MINIMUM_SAMPLE_COUNT


def default_hip_probe(
    backend: HipCommandBackend | None = None,
    architecture: str | None = None,
    *,
    allow_isa_download: bool = True,
) -> HipProbe:
    """Return the concrete HIP probe without resolving HIP tools at import time."""
    backend = backend or HipCommandBackend(
        architecture=architecture, allow_isa_download=allow_isa_download
    )
    return HipProbe(
        compile_candidate=backend.compile,
        execute_candidate=backend.execute,
        check_correctness=backend.correctness,
        check_stability=backend.stability,
        evidence_for=backend.evidence,
    )


def _is_supported_probe(key: CalibrationProfileKey) -> bool:
    if key.operation == "matrix":
        return (
            key.input_dtype in {"bf16", "fp16", "fp32"}
            and key.output_dtype == key.input_dtype
            and key.path in {"wmma", "mfma", "gfx12"}
        )
    if key.operation == "vector":
        return (
            key.input_dtype in {"fp32", "fp16", "bf16"}
            and key.output_dtype == key.input_dtype
        )
    if key.operation == "stream_copy":
        return (key.input_dtype, key.output_dtype) in {
            ("fp32", "fp32"),
            ("fp16", "fp16"),
            ("bf16", "bf16"),
            ("bf16", "fp32"),
            ("fp32", "bf16"),
        }
    return (
        key.kind == "compute"
        and key.operation in {"reduction", "transcendental"}
        and key.input_dtype == key.output_dtype == "fp32"
    )


def _hip_source(key: CalibrationProfileKey) -> str:
    """Load the self-checking HIP benchmark for a calibration candidate."""
    route = (
        key.operation,
        key.input_dtype,
        key.output_dtype,
        key.path if key.operation == "matrix" else None,
    )
    try:
        filename = _PROBE_RESOURCES[route]
    except KeyError as exc:
        raise ValueError(f"unsupported HIP calibration probe: {key.value}") from exc
    return _read_probe_source(filename)


def _read_probe_source(filename: str) -> str:
    """Read one UTF-8 HIP package resource with a consistent failure mode."""
    resource = resources.files(_PROBE_RESOURCE_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise RuntimeError(f"missing HIP calibration probe resource: {filename}")
    try:
        return resource.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"cannot read HIP calibration probe resource: {filename}"
        ) from exc


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _isa_error_code(exc: Exception) -> str:
    name = type(exc).__name__
    return {
        "IsaSpecUnavailableError": "isa_spec_unavailable",
        "IsaDownloadError": "isa_download_failed",
        "IsaIntegrityError": "isa_integrity_failed",
        "IsaHelperBuildError": "isa_helper_build_failed",
        "IsaDecodeError": "isa_decode_failed",
        "IsaProtocolError": "isa_protocol_failed",
        "FileNotFoundError": "isa_artifact_tool_unavailable",
        "ValueError": "isa_artifact_invalid",
    }.get(name, "isa_artifact_validation_failed")
