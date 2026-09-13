from dataclasses import dataclass

from openmind.csp.model.problem import Problem
from openmind.predictor.model.transition_model import TransitionModel
from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class Domain:
    """A domain within the agent: where it starts, which actions are legal and what they lead to."""

    name: str
    initial_state: State
    problem: Problem
    transitions: TransitionModel
