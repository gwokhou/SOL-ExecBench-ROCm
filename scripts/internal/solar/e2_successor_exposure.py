"""One-shot holdout commitment and scoped derivation for a frozen E2 campaign."""

import ctypes
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.internal.solar.e2_contracts import base_seed
from scripts.internal.solar.e2_successor_freeze import (
    VerifiedSuccessorFreeze,
    schedule_sha256,
)
from scripts.internal.solar.e2_successor_ledger import (
    Phase,
    PlannedSession,
    SuccessorLedger,
)
from sol_execbench.core.data.json_utils import (
    canonical_json_bytes,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_bytes, sha256_file

COMMITMENT_FIELD = "holdout_commitment_sha256"
COMMITMENT_FILE = "holdout-commitment.json"
_CLOSED_KEY = "holdout-key-after-online.bin"
_COMMITMENT_KIND = "e2-successor-holdout-commitment-v1"
_PR_SET_DUMPABLE = 4


def protect_observation_process() -> None:
    """Keep private descriptor and memory state outside same-UID proc access."""
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = (ctypes.c_int,) + (ctypes.c_ulong,) * 4
    if prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _write_exclusive(path: Path, payload: bytes) -> None:
    fd = os.open(
        path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(fd, "wb") as file:
        file.write(payload)
        file.flush()
        os.fdatasync(file.fileno())
    parent = os.open(path.parent, os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


@dataclass(frozen=True, slots=True, kw_only=True)
class SuccessorHoldout:
    """Custody of a commitment tied to exactly one frozen E0 roster."""

    ledger: SuccessorLedger
    commitment_sha256: str
    _key: bytes = field(repr=False)

    def seed(self, item: PlannedSession) -> int:
        """Derive a per-session seed without passing the key to the evaluator."""
        if self.ledger.manifest["sessions"][item.index] != item.metadata():
            raise ValueError("E2 seed request is outside the committed roster")
        return base_seed(item.session.case, self._key)

    def reveal_after_online_close(self) -> None:
        """Keep audit material only once the online phase has recorded its terminal state."""
        root = self.ledger.root
        online = load_json_value(root / Phase.ONLINE / "phase-result.json")
        if (
            online["state"] not in {"completed", "INCOMPLETE"}
            or online["roster_sha256"] != self.ledger.roster_sha256
            or sha256_file(root / COMMITMENT_FILE) != self.commitment_sha256
        ):
            raise ValueError(
                "E2 holdout cannot be disclosed before the bound online close"
            )
        _write_exclusive(root / _CLOSED_KEY, self._key)


def commit_successor_holdout(
    ledger: SuccessorLedger,
    freeze: VerifiedSuccessorFreeze,
    *,
    authorized: bool,
) -> SuccessorHoldout:
    """Generate exactly one key for this previously verified execution."""
    if authorized is not True:
        raise PermissionError(
            "opening the frozen successor holdout requires explicit authorization"
        )
    if (
        ledger.manifest["freeze_sha256"] != freeze.sha256
        or ledger.root != freeze.activity
        or freeze.schedule_sha256 != schedule_sha256()
        or sha256_file(freeze.path) != freeze.sha256
    ):
        raise ValueError("E2 holdout does not match its verified freeze")
    path = ledger.root / COMMITMENT_FILE
    if any(
        candidate.exists()
        for candidate in (
            path,
            ledger.root / Phase.ONLINE,
            ledger.root / _CLOSED_KEY,
        )
    ):
        raise FileExistsError("E2 holdout draw or online phase already exists")
    key = secrets.token_bytes(32)
    _write_exclusive(
        path,
        canonical_json_bytes(
            {
                "research_format": _COMMITMENT_KIND,
                "key_sha256": sha256_bytes(key),
                "freeze_sha256": freeze.sha256,
                "roster_sha256": ledger.roster_sha256,
                "schedule_sha256": freeze.schedule_sha256,
                "instances_materialized_at_commitment": 0,
                "created_unix": time.time(),
                "performance_authority": False,
            }
        ),
    )
    return SuccessorHoldout(
        ledger=ledger, commitment_sha256=sha256_file(path), _key=key
    )


def verify_worker_commitment(staging: Path, expected_sha256: object) -> None:
    """Reject a held-out worker unless its private stage names a real commitment."""
    if staging.name != "evidence" or staging.parent.parent.name != Phase.ONLINE:
        raise ValueError(
            "E2 held-out worker is outside the committed online session"
        )
    root = staging.parent.parent.parent
    path = root / COMMITMENT_FILE
    if (
        not isinstance(expected_sha256, str)
        or not path.is_file()
        or sha256_file(path) != expected_sha256
    ):
        raise ValueError(
            "E2 held-out worker lacks its committed exposure receipt"
        )
    payload = load_json_value(path)
    if (
        payload["research_format"] != _COMMITMENT_KIND
        or payload["instances_materialized_at_commitment"] != 0
        or payload["roster_sha256"] != sha256_file(root / "roster.json")
    ):
        raise ValueError(
            "E2 held-out worker commitment differs from the frozen roster"
        )
