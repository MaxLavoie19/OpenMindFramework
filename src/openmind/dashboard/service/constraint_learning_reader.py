import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from openmind.dashboard.model.constraint_learning import ConstraintLearning

logger = logging.getLogger(__name__)

#: What every snapshot says, and so what tells one apart from any other file left in the same directory.
#:
#: Read by what it holds rather than by what it is called. A run is told where to write with a flag and the
#: runs compared this week are called `plain`, `windowed`, `sides` and `further`; a reader looking for one
#: fixed name finds the one run nobody is watching.
POSITION = "position"

#: Where processes are looked for, taken as a parameter everywhere so a test can hand over a directory of its
#: own rather than needing a process that really exists.
PROC = Path("/proc")


class ConstraintLearningReader:
    """Where the dashboard reads constraint-learning runs from.

    A run writes what it has got to after every position and the page reads it afresh on every request, so
    neither waits for the other and neither can hold the other up. A run that has not started, or has written
    nothing yet, reads as nothing rather than as an error: there is no page to show, which is itself the news.

    **Every run in the directory, not one of them.** Arms are the way anything here is settled — the same
    walk with a change and without it — so they are launched together and write side by side. Reading a single
    name showed whichever run happened to carry it, which for two days was one that had already finished."""

    def runs(self, directory: Path, proc: Path = PROC) -> tuple[ConstraintLearning, ...]:
        """Every run that has said anything in that directory, the one that spoke most recently first.

        Sorted by what each run says the time was rather than by the file's own, because a run writing to a
        share writes a file whose time belongs to the machine that holds it. Where a run says nothing about
        when, its file's time stands in.

        Each is told whether the process that wrote it is still there, by looking for the process rather than
        by how long ago it last spoke. A run stuck on one position for an hour is still running and should say
        so; a run killed a second ago is not, however fresh its snapshot."""
        found = Path(directory)
        if not found.is_dir():
            return ()
        held = [one for one in (self._read(at) for at in sorted(found.glob("*.json"))) if one is not None]
        held.sort(key=lambda one: one.at, reverse=True)
        return tuple(replace(one, running=self.alive(one.pid, proc)) for one in held)

    def alive(self, pid: int, proc: Path = PROC) -> bool:
        """Whether that process is still there. A run that named no process is not known to be running."""
        return bool(pid) and (Path(proc) / str(pid)).exists()

    def latest(self, directory: Path, run: str = "", proc: Path = PROC) -> ConstraintLearning | None:
        """That run, or the one that spoke most recently, or None where none has said anything."""
        held = self.runs(directory, proc)
        if run:
            return next((one for one in held if one.run == run), None)
        return held[0] if held else None

    def _read(self, where: Path) -> ConstraintLearning | None:
        """What that file says, or None where it is not a snapshot or cannot be read."""
        try:
            held = json.loads(where.read_text())
        except (OSError, ValueError):
            logger.exception("The constraint learning snapshot at %s could not be read", where)
            return None
        if not isinstance(held, dict) or POSITION not in held:
            return None
        return ConstraintLearning(
            position=int(held.get(POSITION, 0)),
            fen=str(held.get("fen", "")),
            picture=str(held.get("picture", "")),
            rules=tuple(str(one) for one in held.get("rules", ())),
            rightly_refused=int(held.get("rightly refused", 0)),
            let_through=int(held.get("let through", 0)),
            wrongly_refused=int(held.get("wrongly refused", 0)),
            rightly_allowed=int(held.get("rightly allowed", 0)),
            seconds=float(held.get("seconds", 0.0)),
            readings=int(held.get("readings", 0)),
            run=where.stem,
            pid=int(held.get("pid", 0) or 0),
            matching=int(held.get("matching", 0)),
            written_by_hand=int(held.get("written by hand", 0)),
            at=str(held.get("at", "")) or self._when(where),
            consequences=tuple(str(one) for one in held.get("consequences", ())),
            sorts=tuple(str(one) for one in held.get("sorts", ())),
            shapes=tuple(str(one) for one in held.get("shapes", ())),
            couplings=tuple(str(one) for one in held.get("couplings", ())),
        )

    def _when(self, where: Path) -> str:
        """When that file was last written, for a run that does not say."""
        try:
            return datetime.fromtimestamp(where.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            return ""
