from dataclasses import dataclass

from openmind.mcts.model.action_sample import ActionSample
from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class PlayedGame:
    """A self-play game: the samples of every search, the positions searched from in order with the search's mean payoff
    for the player to act in each, the final payoffs in the order of the players' names, in a game between arms the
    arm, the signal, each player followed, in the same order (empty otherwise), and the actions played, in order."""

    samples: tuple[ActionSample, ...]
    states: tuple[State, ...]
    search_values: tuple[float, ...]
    payoffs: tuple[float, ...]
    arms: tuple[str, ...] = ()
    actions: tuple[Action, ...] = ()
