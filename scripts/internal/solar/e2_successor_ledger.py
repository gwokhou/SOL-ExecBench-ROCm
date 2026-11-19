"""One-shot E2 attempt receipts with the original frozen denominators."""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import (
    Activity,
    Mode,
    Session,
    schedule,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_file

_ROSTER = "roster.json"
_STARTED = "started.json"
_CLOSED = "closed.json"


class Phase(StrEnum):
    """Execution phases that each get their own once-only receipt directory."""

    ONLINE = "online"
    OFFLINE = "offline"


class AttemptState(StrEnum):
    """A missing final receipt represents an unknown interrupted attempt."""

    NOT_REACHED = "not_reached"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedSession:
    """Stable index and canonical E0 coordinates for one evaluator session."""

    index: int
    session: Session

    @property
    def directory(self) -> str:
        """Use one stable directory name across both phases."""
        return f"session-{self.index:04d}"

    def metadata(self) -> dict[str, object]:
        """Include the ordinal in the frozen session manifest."""
        return {**self.session.metadata(), "ordinal": self.index}


def planned_sessions(phase: Phase) -> Iterator[PlannedSession]:
    """Filter to incumbent offline work while preserving each original index."""
    for index, session in enumerate(schedule()):
        if phase is Phase.ONLINE or session.mode is Mode.INCUMBENT:
            yield PlannedSession(index=index, session=session)


def _roster(freeze_sha256: str) -> dict[str, Any]:
    rows = [item.metadata() for item in planned_sessions(Phase.ONLINE)]
    units = {
        (row["case_id"], row["strategy"])
        for row in rows
        if row["activity"] == Activity.CORRECTNESS
    }
    return {
        "freeze_sha256": freeze_sha256,
        "performance_authority": False,
        "logical_units": len(units),
        "correctness_sessions": sum(
            row["activity"] == Activity.CORRECTNESS for row in rows
        ),
        "cost_sessions": sum(row["activity"] == Activity.COST for row in rows),
        "offline_sessions": sum(row["mode"] == Mode.INCUMBENT for row in rows),
        "sessions": rows,
    }


class SuccessorLedger:
    """Bind outcomes to one full roster while retaining incomplete and unknown work."""

    def __init__(self, root: Path, *, freeze_sha256: str) -> None:
        """Require an existing canonical roster and a matching execution freeze."""
        self.root = root
        self.manifest = load_json_value(root / _ROSTER)
        if self.manifest != _roster(freeze_sha256):
            raise ValueError("E2 frozen roster or execution identity changed")
        self.roster_sha256 = sha256_file(root / _ROSTER)

    @classmethod
    def create(cls, root: Path, *, freeze_sha256: str) -> SuccessorLedger:
        """Write the denominator in a fresh root before any input can be generated."""
        root.mkdir(exist_ok=False)
        atomic_write_json_value(root / _ROSTER, _roster(freeze_sha256))
        return cls(root, freeze_sha256=freeze_sha256)

    def begin_phase(self, phase: Phase) -> None:
        """Open one new phase. A prior attempt always keeps its directory."""
        folder = self.root / phase
        folder.mkdir(exist_ok=False)
        atomic_write_json_value(
            folder / _STARTED,
            {
                "phase": phase.value,
                "roster_sha256": self.roster_sha256,
                "started_unix": time.time(),
            },
        )

    def _binding(self, phase: Phase, item: PlannedSession) -> dict[str, object]:
        if self.manifest["sessions"][item.index] != item.metadata():
            raise ValueError("E2 attempt is outside the frozen roster")
        if phase is Phase.OFFLINE and item.session.mode is not Mode.INCUMBENT:
            raise ValueError(
                "only incumbent cost sessions have offline replays"
            )
        return {
            "phase": phase.value,
            "session_id": item.session.identity,
            "ordinal": item.index,
            "roster_sha256": self.roster_sha256,
        }

    def start(self, phase: Phase, item: PlannedSession) -> None:
        """Durably record an attempt before invoking the evaluator or oracle."""
        binding = self._binding(phase, item)
        ancestor = self.root / phase
        header = load_json_value(ancestor / _STARTED)
        if (
            header["roster_sha256"] != self.roster_sha256
            or header["phase"] != phase
        ):
            raise ValueError("E2 phase identity changed")
        folder = ancestor / item.directory
        folder.mkdir(exist_ok=False)
        atomic_write_json_value(
            folder / _STARTED,
            {
                **binding,
                "started_unix": time.time(),
            },
        )

    def finish(
        self,
        phase: Phase,
        item: PlannedSession,
        *,
        verdict: str | None,
        stop_reason: str | None,
        evidence: Path | None,
    ) -> None:
        """Preserve numeric rejections as completed outcomes when evidence exists."""
        folder = self.root / phase / item.directory
        binding = self._binding(phase, item)
        started = load_json_value(folder / _STARTED)
        if any(started[key] != value for key, value in binding.items()):
            raise ValueError("E2 attempt receipt identity changed")
        if not stop_reason and evidence is None:
            raise ValueError("a completed E2 attempt requires bound evidence")
        relative = None
        digest = None
        if evidence is not None:
            evidence = evidence.resolve(strict=True)
            if not evidence.is_relative_to(folder.resolve()):
                raise ValueError(
                    "attempt evidence has moved outside its session directory"
                )
            relative = str(evidence.relative_to(folder.resolve()))
            digest = sha256_file(evidence)
        path = folder / _CLOSED
        if path.exists():
            raise FileExistsError(path)
        atomic_write_json_value(
            path,
            {
                **binding,
                "state": AttemptState.INCOMPLETE
                if stop_reason
                else AttemptState.COMPLETED,
                "verdict": verdict,
                "stop_reason": stop_reason,
                "evidence_path": relative,
                "evidence_sha256": digest,
                "finished_unix": time.time(),
            },
        )

    def reconcile(self, phase: Phase) -> list[dict[str, Any]]:
        """Read the entire denominator, keeping not-reached and interrupted rows."""
        rows = []
        for item in planned_sessions(phase):
            path = self.root / phase / item.directory
            metadata = item.metadata()
            binding = self._binding(phase, item)
            observed = {"state": AttemptState.NOT_REACHED, "verdict": None}
            if path.exists():
                observed = {"state": AttemptState.INTERRUPTED, "verdict": None}
                for name in (_STARTED, _CLOSED):
                    receipt_path = path / name
                    if receipt_path.exists():
                        receipt = load_json_value(receipt_path)
                        if any(
                            receipt[key] != value
                            for key, value in binding.items()
                        ):
                            raise ValueError(
                                f"E2 receipt identity changed: {receipt_path}"
                            )
                        if name == _CLOSED:
                            if not (path / _STARTED).is_file():
                                raise ValueError(
                                    "E2 final receipt has no start receipt"
                                )
                            relative = receipt["evidence_path"]
                            if relative is not None:
                                evidence = (path / relative).resolve(
                                    strict=True
                                )
                                if (
                                    not evidence.is_relative_to(path.resolve())
                                    or sha256_file(evidence)
                                    != receipt["evidence_sha256"]
                                ):
                                    raise ValueError(
                                        "E2 bound attempt evidence changed"
                                    )
                            observed = receipt
            rows.append({**metadata, **observed})
        return rows
