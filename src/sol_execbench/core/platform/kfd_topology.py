# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""PCI and KFD topology unique_id resolution for single and partitioned GPUs."""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path

DEFAULT_PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
DEFAULT_KFD_TOPOLOGY_NODES_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")
_HEX_UNIQUE_ID_PATTERN = re.compile(r"[0-9a-f]{16}")


def effective_kfd_nodes_root(
    pci_root: Path, kfd_root: Path | None = None
) -> Path | None:
    """Avoid leaking real host KFD sysfs when tests stub only PCI_DEVICES_ROOT."""
    if kfd_root is not None:
        if (
            pci_root != DEFAULT_PCI_DEVICES_ROOT
            and kfd_root == DEFAULT_KFD_TOPOLOGY_NODES_ROOT
        ):
            return None
        return kfd_root
    if pci_root != DEFAULT_PCI_DEVICES_ROOT:
        return None
    return DEFAULT_KFD_TOPOLOGY_NODES_ROOT


def read_pci_unique_id(pci_root: Path, bdf: str) -> str | None:
    """Read 16-hex ROCr unique_id from PCI sysfs for an unpartitioned BDF."""
    path = pci_root / bdf.strip().lower() / "unique_id"
    with suppress(OSError):
        uid = path.read_text(encoding="utf-8").strip().lower()
        if _HEX_UNIQUE_ID_PATTERN.fullmatch(uid) is not None:
            return uid
    return None


def _parse_kfd_node_properties(path: Path) -> tuple[str, str] | None:
    values: dict[str, str] = {}
    with suppress(OSError):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                values[parts[0]] = parts[1]
    if not values:
        return None
    try:
        if int(values.get("simd_count", "0"), 0) <= 0:
            return None
        domain = int(values["domain"], 0)
        location_id = int(values["location_id"], 0)
    except (KeyError, ValueError):
        return None
    bdf = (
        f"{domain:04x}:{(location_id >> 8) & 0xFF:02x}:"
        f"{(location_id >> 3) & 0x1F:02x}.{location_id & 0x7}"
    )
    raw_uid = values.get("unique_id", "").strip().lower()
    if not raw_uid:
        return None
    try:
        if raw_uid.isdigit():
            parsed_int = int(raw_uid, 10)
            if parsed_int <= 0:
                return None
            uid = f"{parsed_int:016x}"
        elif raw_uid.startswith("0x"):
            parsed_int = int(raw_uid, 16)
            if parsed_int <= 0:
                return None
            uid = f"{parsed_int:016x}"
        else:
            uid = raw_uid
    except ValueError:
        return None
    if _HEX_UNIQUE_ID_PATTERN.fullmatch(uid) is None:
        return None
    return bdf, uid


def read_kfd_gpu_nodes(kfd_root: Path | None) -> tuple[tuple[str, str], ...]:
    """Return ordered (bdf, unique_id_hex) pairs across KFD GPU topology nodes."""
    if kfd_root is None or not kfd_root.is_dir():
        return ()
    node_dirs: list[tuple[int, Path]] = []
    with suppress(OSError):
        for child in kfd_root.iterdir():
            if child.name.isdigit():
                node_dirs.append((int(child.name), child))
    results: list[tuple[str, str]] = []
    for _, node_dir in sorted(node_dirs):
        parsed = _parse_kfd_node_properties(node_dir / "properties")
        if parsed is not None:
            results.append(parsed)
    return tuple(results)


def resolve_bdf_partition_unique_ids(
    bdf: str,
    partition_count: int,
    *,
    pci_root: Path,
    kfd_root: Path | None = None,
) -> tuple[str, ...]:
    """Resolve 1-to-1 ordered ROCr unique_ids for single or partitioned BDFs."""
    normalized_bdf = bdf.strip().lower()
    if partition_count <= 0:
        return ()
    effective_kfd = effective_kfd_nodes_root(pci_root, kfd_root)
    if partition_count == 1:
        pci_uid = read_pci_unique_id(pci_root, normalized_bdf)
        if pci_uid is not None:
            return (pci_uid,)
    kfd_uids = tuple(
        uid
        for node_bdf, uid in read_kfd_gpu_nodes(effective_kfd)
        if node_bdf == normalized_bdf
    )
    if len(kfd_uids) == partition_count and len(set(kfd_uids)) == len(kfd_uids):
        return kfd_uids
    return ()
