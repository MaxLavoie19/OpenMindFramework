from dataclasses import dataclass

from openmind.inference.model.evidence import Evidence


@dataclass(frozen=True, slots=True)
class Scouted:
    """One position somebody looked at and thought worth learning from, and how wrong the rules were there.

    **Small on purpose.** This is what crosses between a process that looks and a process that learns, so it
    carries a board and the actions the game lists and nothing else. The thousands of candidates a learner reads
    are worked out from these two, by whoever is going to read them — sending them would be sending two million
    readings to save reading a board.

    `wrongly` is why it was picked: how many of the moves the game allows the constraints turned away. It is
    kept rather than recomputed because it is what the positions are ranked by, and because it was measured
    against the constraints *as they stood when it was scouted*, which is a fact about this scouting and not
    about the position."""

    where: Evidence
    wrongly: int
