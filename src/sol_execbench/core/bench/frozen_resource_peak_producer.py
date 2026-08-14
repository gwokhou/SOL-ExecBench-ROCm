# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Typed adapter for the immutable RDNA4 resource-peak v3 producer.

The historical script is evidence-bound and must remain byte-for-byte stable.
This module contains the single compatibility seam that loads it and exposes a
small typed API to the qualified successor. New producers must implement a
normal importable API instead of extending this adapter.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

type ResourcePeakProbe = Mapping[str, object]


class ResourcePeakTuning(Protocol):
    """Fields consumed from one frozen probe tuning specification."""

    compiler_macro: str
    candidates: tuple[int, ...]


class ResourcePeakSampleBatch(Protocol):
    """Serializable sample batch returned by the frozen producer."""

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible batch payload."""
        ...


class _CompileProbe(Protocol):
    def __call__(
        self,
        source: Path,
        output_dir: Path,
        hipcc: str,
        gfx: str,
        *,
        compiler_defines: Mapping[str, int] | None = None,
    ) -> Path:
        """Compile one probe with optional frozen tuning definitions."""
        ...


class _RunSampleBatch(Protocol):
    def __call__(
        self,
        binary: Path,
        process_batch: int,
        *,
        amdsmi: object | None,
    ) -> ResourcePeakSampleBatch:
        """Run one sample batch through the historical producer."""
        ...


@dataclass(frozen=True, slots=True)
class FrozenResourcePeakProducer:
    """Public typed view of the evidence-bound historical producer."""

    source_path: Path
    probe_dir: Path
    probes: tuple[ResourcePeakProbe, ...]
    _compile: _CompileProbe
    _sample_batch: _RunSampleBatch
    _clock: object
    _required_tool: object
    _device: object
    _revision: object
    _run_calibration: object

    def compile_probe(
        self,
        source: Path,
        output_dir: Path,
        hipcc: str,
        gfx: str,
        *,
        compiler_defines: Mapping[str, int] | None = None,
    ) -> Path:
        """Compile one frozen probe through the historical implementation."""
        return self._compile(
            source,
            output_dir,
            hipcc,
            gfx,
            compiler_defines=compiler_defines,
        )

    def run_sample_batch(
        self,
        binary: Path,
        process_batch: int,
    ) -> ResourcePeakSampleBatch:
        """Run one minimal sample batch without AMD SMI telemetry."""
        return self._sample_batch(binary, process_batch, amdsmi=None)

    def clock_state(self) -> Mapping[str, object]:
        """Return the historical producer's clock observation."""
        function = cast(Callable[[], Mapping[str, object]], self._clock)
        return function()

    def required_rocm_tool(self, name: str) -> str:
        """Resolve a required ROCm tool through the frozen producer."""
        function = cast(Callable[[str], str], self._required_tool)
        return function(name)

    def device_identity(self) -> dict[str, object]:
        """Return the frozen producer's device identity."""
        function = cast(Callable[[], dict[str, object]], self._device)
        return function()

    def git_revision(self) -> str:
        """Return the source revision observed by the frozen producer."""
        function = cast(Callable[[], str], self._revision)
        return function()

    def calibrate(self, arguments: argparse.Namespace, workdir: Path) -> int:
        """Run the historical calibration implementation."""
        function = cast(
            Callable[[argparse.Namespace, Path], int],
            self._run_calibration,
        )
        return function(arguments, workdir)


def _required_attribute(module: ModuleType, name: str) -> object:
    try:
        return getattr(module, name)
    except AttributeError as error:
        raise ImportError(
            f"frozen resource-peak producer lacks required symbol {name}",
        ) from error


@cache
def load_frozen_resource_peak_producer(
    source_path: Path,
) -> FrozenResourcePeakProducer:
    """Load and validate the one immutable historical producer module."""
    path = source_path.resolve()
    module_name = "_sol_execbench_frozen_rdna4_resource_peak_v3"
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen resource-peak producer: {path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    probes = _required_attribute(module, "PROBES")
    probe_dir = _required_attribute(module, "PROBE_DIR")
    if not isinstance(probes, tuple) or not isinstance(probe_dir, Path):
        raise ImportError("frozen resource-peak producer contract is invalid")
    return FrozenResourcePeakProducer(
        source_path=path,
        probe_dir=probe_dir,
        probes=cast(tuple[ResourcePeakProbe, ...], probes),
        _compile=cast(
            _CompileProbe, _required_attribute(module, "_compile_probe")
        ),
        _sample_batch=cast(
            _RunSampleBatch,
            _required_attribute(module, "_run_sample_batch"),
        ),
        _clock=_required_attribute(module, "_clock_state"),
        _required_tool=_required_attribute(module, "_required_rocm_tool"),
        _device=_required_attribute(module, "_device_identity"),
        _revision=_required_attribute(module, "_git_revision"),
        _run_calibration=_required_attribute(module, "_calibrate"),
    )


__all__ = [
    "FrozenResourcePeakProducer",
    "ResourcePeakProbe",
    "ResourcePeakSampleBatch",
    "ResourcePeakTuning",
    "load_frozen_resource_peak_producer",
]
