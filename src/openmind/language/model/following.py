from dataclasses import dataclass

from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Following:
    """How far a game somebody else wrote down could be followed, and where it stopped.

    `positions` are the positions it went through, the first being where it started, so there is one more of
    them than there are `actions` — until it stopped, where there are as many as it managed.

    `stopped` says why it went no further, and empty means it followed the whole thing. **Why it stopped is
    the news, not a failure to report.** A notation that narrowed to nothing says the rules refuse a move
    somebody really played, which is a rule too tight and the one mistake nothing else reports. A notation
    that narrowed to several says the reading cannot tell those moves apart yet, which is a coupling not
    learned. The two want opposite work and a bare count of positions would hide which it was."""

    positions: tuple[State, ...]
    actions: tuple[Action, ...]
    read: int
    stopped: str = ""

    @property
    def whole(self) -> bool:
        """Whether every notation given was followed."""
        return not self.stopped

    @property
    def followed(self) -> int:
        """How many of the notations were read down to one move."""
        return len(self.actions)
