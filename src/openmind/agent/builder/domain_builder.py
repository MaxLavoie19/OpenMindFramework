from typing import Self

from openmind.agent.model.domain import Domain
from openmind.csp.model.problem import Problem
from openmind.predictor.model.transition_model import TransitionModel
from openmind.world.model.state import State


class DomainBuilder:
    """Collects the parts of a domain within the agent."""

    def __init__(self) -> None:
        self._name: str | None = None
        self._initial_state: State | None = None
        self._problem: Problem | None = None
        self._transitions: TransitionModel | None = None

    def with_name(self, name: str) -> Self:
        self._name = name
        return self

    def with_initial_state(self, state: State) -> Self:
        self._initial_state = state
        return self

    def with_problem(self, problem: Problem) -> Self:
        self._problem = problem
        return self

    def with_transitions(self, transitions: TransitionModel) -> Self:
        self._transitions = transitions
        return self

    def build(self) -> Domain:
        name, initial_state, problem, transitions = (
            self._name,
            self._initial_state,
            self._problem,
            self._transitions,
        )
        if name is None or initial_state is None or problem is None or transitions is None:
            parts = (("name", name), ("initial state", initial_state), ("problem", problem), ("transitions", transitions))
            missing = ", ".join(part for part, value in parts if value is None)
            raise ValueError(f"Domain is missing: {missing}")
        return Domain(name, initial_state, problem, transitions)
