from dataclasses import dataclass

from openmind.inference.model.deduction import Deduction
from openmind.training.model.played_game import PlayedGame


@dataclass(frozen=True, slots=True)
class GameLesson:
    """What a worker studied of one game it played: the game, and the deductions of its positions walked back from its
    end, the last position first, empty for a game not walked back."""

    game: PlayedGame
    walk: tuple[Deduction, ...] = ()
