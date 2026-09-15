from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalReadings:
    """What signals read on a round's anchors: how many anchors there were, how many took their winner from a proof, how
    many games and decisive games they came from, and by signal name, at every anchor, +1 where the signal pointed to
    the winner, -1 to the loser and 0 for a tie, None for a signal that couldn't be read."""

    anchors: int
    proven: int
    games: int
    decisive: int
    signs: dict[str, np.ndarray | None]
