from dataclasses import dataclass

from openmind.mcts.model.action_sample import ActionSample
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.world.model.action import Action
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class PlayedGame:
    """A self-play game: the samples of every search, the positions searched from in order with the search's mean payoff
    for the player to act in each, the final payoffs in the order of the players' names, in a game between arms the
    arm each player followed, in the same order (empty otherwise), and the actions played, in order. On a
    clock: the time control, each step's seconds and budget in order (a step that ran its player's time out included,
    though its action wasn't played), each player's clock at the end in the order of the players' names, and the player
    whose time ran out, if any. Then the seeds the game was played from and why it ended, when the domain says or a
    player's time ran out."""

    samples: tuple[ActionSample, ...]
    states: tuple[State, ...]
    search_values: tuple[float, ...]
    payoffs: tuple[float, ...]
    arms: tuple[str, ...] = ()
    actions: tuple[Action, ...] = ()
    time_control: TimeControl | None = None
    seconds: tuple[float, ...] = ()
    budgets: tuple[float | None, ...] = ()
    clocks: tuple[Clock, ...] = ()
    flagged: str | None = None
    agent_seed: int | None = None
    outcome_seed: int | None = None
    ending: str | None = None
