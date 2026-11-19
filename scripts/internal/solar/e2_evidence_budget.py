"""Evidence admission at quiescent boundaries of a serial E2 activity."""

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from scripts.internal.solar.e2_contracts import EVIDENCE_BUDGET_BYTES

_STAT_BLOCK_BYTES = 512
ADMINISTRATIVE_RESERVE_BYTES = 64 * 1024**2
OFFLINE_PENDING_ARCHIVE_BYTES = 48 * 1024**2


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceUsage:
    """Named logical payload and allocated storage, including directories."""

    logical_bytes: int
    allocated_bytes: int


class EvidenceBudgetError(RuntimeError):
    """The next writer and the stop receipt cannot both be budgeted."""


def retained_evidence(root: Path) -> EvidenceUsage:
    """Measure a drained evidence tree on one filesystem.

    An anonymous spool must have closed or linked every descriptor before
    this check. Reject links to other filesystems and non-file artifacts.
    Internal hard links are counted once as they share one allocation.
    """
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("E2 evidence root must be a directory")
    device = root_stat.st_dev
    allocated = root_stat.st_blocks * _STAT_BLOCK_BYTES
    logical = 0
    pending = [root]
    linked_files: dict[int, int] = {}
    link_totals: dict[int, int] = {}
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if info.st_dev != device:
                    raise ValueError(
                        f"E2 evidence crossed a filesystem: {entry.path}"
                    )
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    if info.st_nlink > 1:
                        inode = info.st_ino
                        count = linked_files.get(inode, 0)
                        linked_files[inode] = count + 1
                        link_totals[inode] = info.st_nlink
                        if count:
                            continue
                    logical += info.st_size
                else:
                    raise ValueError(
                        f"unexpected E2 evidence entry: {entry.path}"
                    )
                allocated += info.st_blocks * _STAT_BLOCK_BYTES
    if linked_files != link_totals:
        raise ValueError("E2 evidence is hard linked outside the activity root")
    return EvidenceUsage(logical_bytes=logical, allocated_bytes=allocated)


def admit_evidence_writer(
    root: Path,
    *,
    writer_limit_bytes: int,
    limit_bytes: int = EVIDENCE_BUDGET_BYTES,
) -> EvidenceUsage:
    """Require a complete writer envelope before starting or materializing inputs.

    The writer must independently enforce its bounded allocation, including
    anonymous and pending files. Only one such writer can run at a time.
    The caller must verify usage again once it is drained.
    """
    if not 0 < writer_limit_bytes <= limit_bytes <= EVIDENCE_BUDGET_BYTES:
        raise ValueError("invalid E2 evidence limits")
    retained = retained_evidence(root)
    required = writer_limit_bytes + ADMINISTRATIVE_RESERVE_BYTES
    if (
        max(retained.logical_bytes, retained.allocated_bytes) + required
        > limit_bytes
    ):
        raise EvidenceBudgetError(
            "evidence budget cannot accommodate next writer and closeout"
        )
    usage = os.statvfs(root)
    if required > usage.f_bavail * usage.f_frsize:
        raise EvidenceBudgetError(
            "filesystem lacks space for next writer and closeout"
        )
    return retained


def admit_offline_writer(
    root: Path, *, limit_bytes: int = EVIDENCE_BUDGET_BYTES
) -> int:
    """Return the drainable byte allowance for one serial offline replay.

    The admitted replay charges completed calls against this allowance. Only
    one pending archive can be allocated, and the writer bounds it before
    creating its pending directory. Published calls reuse that allocation.
    """
    retained = admit_evidence_writer(
        root,
        writer_limit_bytes=OFFLINE_PENDING_ARCHIVE_BYTES,
        limit_bytes=limit_bytes,
    )
    disk = os.statvfs(root)
    allowance = (
        min(
            limit_bytes - max(retained.logical_bytes, retained.allocated_bytes),
            disk.f_bavail * disk.f_frsize,
        )
        - ADMINISTRATIVE_RESERVE_BYTES
    )
    if allowance < OFFLINE_PENDING_ARCHIVE_BYTES:
        raise EvidenceBudgetError("filesystem lost its offline write allowance")
    return allowance
