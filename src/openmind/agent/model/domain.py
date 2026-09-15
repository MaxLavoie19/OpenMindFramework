from dataclasses import dataclass

from openmind.csp.model.problem import Problem
from openmind.observation.model.observation import Observation
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.rule import Rule
from openmind.world.model.players import Players
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Domain:
    """A domain within the agent: where it starts, which actions are legal, what they lead to, who plays, what each
    player sees of a state (None: every player sees everything), and, for the logs, why a finished game ended and its
    record: `ending`, a rule reading a finished game's last state, and `record`, a rule reading the initial state and the
    parameter `actions`, both seeing the transitions' definitions when they are source (None: not said). Every rule is
    Python source or one of the project's own functions."""

    name: str
    initial_state: State
    problem: Problem
    transitions: TransitionModel
    players: Players
    observation: Observation | None = None
    ending: Rule | None = None
    record: Rule | None = None
