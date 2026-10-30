# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Read raw AMD-SMI throttle evidence without reducing invalid fields to bool."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

from sol_execbench.core.integrity.checksums import sha256_file
from sol_execbench.core.process.logs import redacted_text_tail
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

QUERY_TIMEOUT_SECONDS = 10.0
MAX_CAPTURE_BYTES = 16 * 1024


def _load_api(amdsmi: Path) -> ModuleType:
    executable = amdsmi.resolve(strict=True)
    if executable.parent.name == "bin":
        root = executable.parent.parent
    elif executable.parent.name == "amdsmi_cli":
        root = executable.parent.parent.parent
    else:
        raise ValueError("unknown AMD-SMI installation layout")
    package = root / "share" / "amd_smi"
    if not (package / "amdsmi" / "__init__.py").is_file():
        raise FileNotFoundError(
            "AMD-SMI binding is missing beside selected CLI"
        )
    sys.path.insert(0, str(package))
    return importlib.import_module("amdsmi.amdsmi_interface")


def _raw_snapshot(api: ModuleType, handle: object) -> dict[str, object]:
    binding_file = api.__file__
    if binding_file is None:
        raise RuntimeError("AMD-SMI binding has no source identity")
    wrapper = api.amdsmi_wrapper
    metrics = wrapper.amdsmi_gpu_metrics_t()
    result = wrapper.amdsmi_get_gpu_metrics_info(handle, ctypes.byref(metrics))
    if result != wrapper.AMDSMI_STATUS_SUCCESS:
        raise RuntimeError(f"AMD-SMI raw metrics query failed: {result}")
    fields = {}
    for name, limit in (
        ("throttle_status", api.MaxUIntegerTypes.UINT32_T),
        ("indep_throttle_status", api.MaxUIntegerTypes.UINT64_T),
    ):
        raw = getattr(metrics, name)
        converted = api._validate_if_max_uint(raw, limit, isBool=True)
        fields[name] = {
            "value": raw,
            "type": type(raw).__name__,
            "hex": hex(raw),
            "binding_value": converted,
            "binding_type": type(converted).__name__,
        }
    return {
        "raw": fields,
        "metrics_header": {
            name: getattr(metrics.common_header, name)
            for name in (
                "format_revision",
                "content_revision",
                "structure_size",
            )
        },
        "api_version": api.amdsmi_get_lib_version(),
        "binding_path": api.__file__,
        "binding_sha256": sha256_file(Path(binding_file)),
        "uuid": api.amdsmi_get_gpu_device_uuid(handle),
        "bdf": api.amdsmi_get_gpu_device_bdf(handle),
    }


def collect_throttle_evidence(amdsmi: Path, index: int) -> dict[str, object]:
    """Management reads only; the caller owns the native GPU lifecycle."""
    version = run_in_process_group_bounded(
        [str(amdsmi), "version"],
        timeout=QUERY_TIMEOUT_SECONDS,
        max_capture_bytes=MAX_CAPTURE_BYTES,
    )
    if version.returncode != 0 or not version.stdout.strip():
        raise RuntimeError("AMD-SMI CLI version query failed")
    api = _load_api(amdsmi)
    api.amdsmi_init()
    try:
        handles = api.amdsmi_get_processor_handles()
        if index < 0 or index >= len(handles):
            raise ValueError("AMD-SMI device index is unavailable")
        result = _raw_snapshot(api, handles[index])
        result["cli_version"] = version.stdout.strip()[:1000]
        result["cli_path"] = str(amdsmi.resolve())
        return result
    finally:
        api.amdsmi_shut_down()


def main() -> None:
    """Emit one bounded child-process management observation as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amdsmi", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    try:
        result = collect_throttle_evidence(args.amdsmi, args.gpu)
    except Exception as exc:  # noqa: BLE001 -- retain failed management observations as unknown
        result = {
            "error": f"{type(exc).__name__}: {redacted_text_tail(str(exc))}"
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
