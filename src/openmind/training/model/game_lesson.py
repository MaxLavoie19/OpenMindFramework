from dataclasses import dataclass

import numpy as np

from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.pondering import Pondering
from openmind.training.model.signal import Signal
from openmind.training.model.signal_readings import SignalReadings


@dataclass(frozen=True, slots=True)
class GameLesson:
    """What a worker learned from one game it played: the game; every candidate signal, read on the game's anchors; what
    pondering the game gave (None without pondering), its proofs, its walk back from the end and the seeds the proofs
    induced; by followed signal, the targets of every position of the game for each player, in the order
    `SignalTargeter.rows` gives them; and by term, what each term of the arms' value rules read on those rows, None for
    a term that couldn't be read on every row."""

    game: PlayedGame
    candidates: tuple[Signal, ...]
    readings: SignalReadings
    pondering: Pondering | None
    targets: dict[str, np.ndarray]
    terms: dict[PythonRule, np.ndarray | None]
