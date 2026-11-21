# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Boundary-only single-GPU defaults for host CLI and Docker launchers."""

from __future__ import annotations

import argparse
import os
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from sol_execbench.core.bench.clock_ownership import (
    ClockDevice,
    observe_clock_devices,
)
from sol_execbench.core.bench.gpu_admission import CLOCK_STATE_ENV
from sol_execbench.core.bench.gpu_lock_paths import (
    GPU_LOCK_DIR_ENV,
    GPU_LOCK_FD_ENV,
    GPU_LOCK_HOST_ENV,
    GPU_LOCK_UUID_ENV,
    LEGACY_GPU_LOCK_ENV,
    configured_gpu_locks,
    normalize_gpu_uuid,
    prepare_gpu_locks,
)
from sol_execbench.core.bench.rocm_profiler.hip_scope_models import (
    ROCR_UUID_PATTERN,
)
from sol_execbench.core.platform.kfd_topology import (
    DEFAULT_KFD_TOPOLOGY_NODES_ROOT,
    DEFAULT_PCI_DEVICES_ROOT,
    effective_kfd_nodes_root,
    read_kfd_gpu_nodes,
    resolve_bdf_partition_unique_ids,
)
from sol_execbench.core.process.environment import (
    ENV_ROCR_VISIBLE_DEVICES,
    ENV_SOL_EXECBENCH_SANDBOXED,
)

PCI_DEVICES_ROOT = DEFAULT_PCI_DEVICES_ROOT
KFD_TOPOLOGY_NODES_ROOT = DEFAULT_KFD_TOPOLOGY_NODES_ROOT
_ENV_LOCK = threading.Lock()
_ACTIVE_KEY_COUNTS: Counter[str] = Counter()


def _device_unique_id_pairs(
    devices: Sequence[ClockDevice],
    pci_root: Path,
    kfd_root: Path | None,
) -> list[tuple[ClockDevice, str]]:
    grouped: dict[str, list[ClockDevice]] = {}
    for device in devices:
        grouped.setdefault(device.bdf.strip().lower(), []).append(device)
    pairs: list[tuple[ClockDevice, str]] = []
    for bdf, members in grouped.items():
        uids = resolve_bdf_partition_unique_ids(
            bdf, len(members), pci_root=pci_root, kfd_root=kfd_root
        )
        if len(uids) == len(members):
            pairs.extend(zip(members, uids, strict=True))
    return pairs


def _has_topology_sources(pci_root: Path, kfd_root: Path | None) -> bool:
    if any(pci_root.glob("*/unique_id")):
        return True
    effective_kfd = effective_kfd_nodes_root(pci_root, kfd_root)
    return bool(read_kfd_gpu_nodes(effective_kfd))


def _derive_gpu_identity(
    env: Mapping[str, str],
    pci_root: Path,
    kfd_root: Path | None,
    device_observer: Callable[[], tuple[ClockDevice, ...]],
) -> dict[str, str]:
    has_uuid = GPU_LOCK_UUID_ENV in env
    has_visible = ENV_ROCR_VISIBLE_DEVICES in env
    if (has_uuid and has_visible) or not _has_topology_sources(
        pci_root, kfd_root
    ):
        return {}
    derived: dict[str, str] = {}
    with suppress(Exception):
        pairs = _device_unique_id_pairs(device_observer(), pci_root, kfd_root)
        if not has_uuid and not has_visible:
            if len(pairs) == 1:
                sole_device, sole_uid = pairs[0]
                derived[GPU_LOCK_UUID_ENV] = sole_device.uuid.lower()
                derived[ENV_ROCR_VISIBLE_DEVICES] = f"GPU-{sole_uid}"
        elif has_visible and not has_uuid:
            visible = env[ENV_ROCR_VISIBLE_DEVICES].strip()
            if re.fullmatch(ROCR_UUID_PATTERN, visible) is not None:
                want_uid = visible.removeprefix("GPU-").lower()
                matched = [dev for dev, uid in pairs if uid == want_uid]
                if len(matched) == 1:
                    derived[GPU_LOCK_UUID_ENV] = matched[0].uuid.lower()
        elif has_uuid and not has_visible:
            want_uuid = normalize_gpu_uuid(env[GPU_LOCK_UUID_ENV])
            matched_uids = [
                uid for dev, uid in pairs if dev.uuid.lower() == want_uuid
            ]
            if len(matched_uids) == 1:
                derived[ENV_ROCR_VISIBLE_DEVICES] = f"GPU-{matched_uids[0]}"
    return derived


def derive_single_gpu_host_defaults(
    environment: Mapping[str, str] | None = None,
    *,
    pci_root: Path | None = None,
    kfd_root: Path | None = None,
    device_observer: Callable[[], tuple[ClockDevice, ...]] | None = None,
) -> dict[str, str]:
    """Derive omitted host lock paths and unique-GPU selectors safely."""
    env = os.environ if environment is None else environment
    if (
        env.get(GPU_LOCK_HOST_ENV) == "1"
        or env.get(ENV_SOL_EXECBENCH_SANDBOXED) == "1"
        or env.get(GPU_LOCK_FD_ENV) is not None
        or any(name in env for name in LEGACY_GPU_LOCK_ENV)
    ):
        return {}

    derived: dict[str, str] = {}
    if GPU_LOCK_DIR_ENV not in env:
        xdg_state = env.get("XDG_STATE_HOME", "").strip()
        lock_root = (
            Path(xdg_state)
            if xdg_state and Path(xdg_state).is_absolute()
            else Path.home() / ".local" / "state"
        )
        lock_dir = str(lock_root)
        derived[GPU_LOCK_DIR_ENV] = lock_dir
    else:
        lock_dir = env[GPU_LOCK_DIR_ENV].strip()

    if CLOCK_STATE_ENV not in env and lock_dir and Path(lock_dir).is_absolute():
        derived[CLOCK_STATE_ENV] = str(Path(lock_dir) / "gpu-leases" / "sol")

    root = PCI_DEVICES_ROOT if pci_root is None else pci_root
    nodes_root = KFD_TOPOLOGY_NODES_ROOT if kfd_root is None else kfd_root
    observer = (
        observe_clock_devices if device_observer is None else device_observer
    )
    derived.update(_derive_gpu_identity(env, root, nodes_root, observer))
    return derived


@contextmanager
def host_single_gpu_defaults() -> Iterator[dict[str, str]]:
    """Temporarily expose derived single-GPU host defaults within a boundary."""
    with _ENV_LOCK:
        baseline = {
            key: value
            for key, value in os.environ.items()
            if key not in _ACTIVE_KEY_COUNTS
        }
    derived = derive_single_gpu_host_defaults(baseline)
    applied: list[str] = []
    if derived:
        with _ENV_LOCK:
            for key, value in derived.items():
                if key not in os.environ or key in _ACTIVE_KEY_COUNTS:
                    os.environ[key] = value
                    _ACTIVE_KEY_COUNTS[key] += 1
                    applied.append(key)
    try:
        yield derived
    finally:
        if applied:
            with _ENV_LOCK:
                for key in applied:
                    _ACTIVE_KEY_COUNTS[key] -= 1
                    if _ACTIVE_KEY_COUNTS[key] <= 0:
                        del _ACTIVE_KEY_COUNTS[key]
                        os.environ.pop(key, None)


def main(argv: Sequence[str] | None = None) -> int:
    """Emit omitted single-GPU host defaults as KEY=VALUE lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-locks", action="store_true")
    args = parser.parse_args(argv)

    derived = derive_single_gpu_host_defaults()
    merged = {**os.environ, **derived}
    if args.prepare_locks:
        if GPU_LOCK_UUID_ENV in derived:
            prepare_gpu_locks(configured_gpu_locks(merged))
        clock_dir = merged.get(CLOCK_STATE_ENV, "").strip()
        if clock_dir and Path(clock_dir).is_absolute():
            private_path = Path(clock_dir)
            private_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            private_path.chmod(0o700)

    for key, value in derived.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"derived {key} contains an invalid newline")
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
