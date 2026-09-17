from typing import Self

from openmind.agent.model.domain import Domain
from openmind.csp.model.problem import Problem
from openmind.observation.model.observation import Observation
from openmind.predictor.model.transition_model import TransitionModel
from openmind.csp.model.state_domain import StateDomain
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.model.rule import Rule
from openmind.world.model.players import Players
from openmind.world.model.state import State


class DomainBuilder:
    """Collects the parts of a domain within the agent. A rule may be Python source or one of the project's own
    functions; a function a worker process couldn't find, such as a lambda or one defined inside another function, is
    rejected as it is given."""

    def __init__(self) -> None:
        self._caller = create_rule_caller()
        self._name: str | None = None
        self._initial_state: State | None = None
        self._problem: Problem | None = None
        self._transitions: TransitionModel | None = None
        self._players: Players | None = None
        self._observation: Observation | None = None
        self._ending: Rule | None = None
        self._record: Rule | None = None
        self._timeout: Rule | None = None
        self._picture: Rule | None = None

    def with_name(self, name: str) -> Self:
        self._name = name
        return self

    def with_initial_state(self, state: State) -> Self:
        self._initial_state = state
        return self

    def with_problem(self, problem: Problem) -> Self:
        for definition in problem.actions:
            for constraint in definition.constraints:
                self._caller.check(constraint)
            for variable in definition.variables:
                if isinstance(variable.domain, StateDomain):
                    self._caller.check(variable.domain.rule)
        self._problem = problem
        return self

    def with_transitions(self, transitions: TransitionModel) -> Self:
        for transition in transitions.transitions:
            for branch in transition.branches:
                self._caller.check(branch.effects)
        for branch in transitions.resolution or ():
            self._caller.check(branch.effects)
        self._transitions = transitions
        return self

    def with_players(self, players: Players) -> Self:
        self._players = players
        return self

    def with_observation(self, observation: Observation | None) -> Self:
        """What each player sees of a state; None, the default, shows every player everything."""
        if observation is not None:
            self._caller.check(observation.hidden)
            self._caller.check(observation.completions)
        self._observation = observation
        return self

    def with_ending(self, ending: Rule | None) -> Self:
        """The rule saying why a finished game ended, reading its last state; None, the default, doesn't say."""
        self._caller.check(ending)
        self._ending = ending
        return self

    def with_record(self, record: Rule | None) -> Self:
        """The rule giving a game's record, reading the initial state and `actions`; None, the default, gives none."""
        self._caller.check(record)
        self._record = record
        return self

    def with_timeout(self, timeout: Rule | None) -> Self:
        """The effects rule giving the state after a player's clock ran out, reading `flagged`, the player's name; None, the
        default, can't be played on a clock."""
        self._caller.check(timeout)
        self._timeout = timeout
        return self

    def with_picture(self, picture: Rule | None) -> Self:
        """The rule drawing a position as an SVG image, reading the state and `last`, the action that led to it; None, the
        default, shows positions as text."""
        self._caller.check(picture)
        self._picture = picture
        return self

    def build(self) -> Domain:
        name, initial_state, problem, transitions, players = (
            self._name,
            self._initial_state,
            self._problem,
            self._transitions,
            self._players,
        )
        if name is None or initial_state is None or problem is None or transitions is None or players is None:
            parts = (
                ("name", name),
                ("initial state", initial_state),
                ("problem", problem),
                ("transitions", transitions),
                ("players", players),
            )
            missing = ", ".join(part for part, value in parts if value is None)
            raise ValueError(f"Domain is missing: {missing}")
        return Domain(name, initial_state, problem, transitions, players, self._observation, self._ending, self._record, self._timeout, self._picture)
