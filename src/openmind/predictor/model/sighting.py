from dataclasses import dataclass

from openmind.predictor.model.watched import Watched


@dataclass(frozen=True, slots=True)
class Sighting:
    """One move seen, and what the game called it.

    **Two things that have to stay in step, kept as one.** What a move did and what it was named are learned by
    different learners from the same moment, and while they travelled as two lists side by side, staying aligned
    was something every caller had to remember to do. Crossing a process boundary is where a pairing kept by
    convention becomes a pairing kept by a type.

    `said` is the game's own name for it and may be empty: a game need not have a notation, and nothing here
    requires one."""

    watched: Watched
    said: str = ""
