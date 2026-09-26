import json
import logging
from pathlib import Path

from openmind.dashboard.model.constraint_learning import ConstraintLearning

logger = logging.getLogger(__name__)

#: What a run learning constraints leaves for the dashboard to read, in the directory it logs to.
SNAPSHOT = "constraints.json"


class ConstraintLearningReader:
    """Where the dashboard reads a constraint-learning run from.

    The run writes what it has got to after every position and the page reads it afresh on every request, so
    neither waits for the other and neither can hold the other up. A run that has not started, or has written
    nothing yet, reads as nothing rather than as an error: there is no page to show, which is itself the news."""

    def latest(self, directory: Path) -> ConstraintLearning | None:
        """What the run has got to, or None where it has not said."""
        found = Path(directory) / SNAPSHOT
        if not found.exists():
            return None
        try:
            held = json.loads(found.read_text())
        except (OSError, ValueError):
            logger.exception("The constraint learning snapshot at %s could not be read", found)
            return None
        return ConstraintLearning(
            position=int(held.get("position", 0)),
            fen=str(held.get("fen", "")),
            picture=str(held.get("picture", "")),
            rules=tuple(str(one) for one in held.get("rules", ())),
            rightly_refused=int(held.get("rightly refused", 0)),
            let_through=int(held.get("let through", 0)),
            wrongly_refused=int(held.get("wrongly refused", 0)),
            rightly_allowed=int(held.get("rightly allowed", 0)),
            seconds=float(held.get("seconds", 0.0)),
            readings=int(held.get("readings", 0)),
            matching=int(held.get("matching", 0)),
            written_by_hand=int(held.get("written by hand", 0)),
            at=str(held.get("at", "")),
            consequences=tuple(str(one) for one in held.get("consequences", ())),
            sorts=tuple(str(one) for one in held.get("sorts", ())),
            shapes=tuple(str(one) for one in held.get("shapes", ())),
            couplings=tuple(str(one) for one in held.get("couplings", ())),
        )
