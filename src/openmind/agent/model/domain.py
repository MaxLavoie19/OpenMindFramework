from dataclasses import dataclass

from openmind.csp.model.problem import Problem
from openmind.observation.model.observation import Observation
from openmind.predictor.model.transition_model import TransitionModel
from openmind.world.model.players import Players
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Domain:
    """A domain within the agent: where it starts, which actions are legal, what they lead to, who plays, and what each
    player sees of a state (None: every player sees everything)."""

    name: str
    initial_state: State
    problem: Problem
    transitions: TransitionModel
    players: Players
    observation: Observation | None = None
