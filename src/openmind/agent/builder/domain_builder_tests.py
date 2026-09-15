import pytest

from openmind.agent.builder.domain_builder import DomainBuilder
from openmind.agent.model.domain import Domain
from openmind.csp.model.problem import Problem
from openmind.observation.model.observation import Observation
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.world.model.players import Players
from openmind.world.model.state import State


def test_build_assembles_the_parts() -> None:
    state = State((("light", "off"), ("payoff", None), ("turn", "me")))
    players = Players(("me",), "turn", ("payoff",))

    domain = (
        DomainBuilder()
        .with_name("light")
        .with_initial_state(state)
        .with_problem(Problem(()))
        .with_transitions(TransitionModel(()))
        .with_players(players)
        .build()
    )

    assert domain == Domain("light", state, Problem(()), TransitionModel(()), players)
    assert domain.observation is None


def test_build_keeps_what_each_player_sees() -> None:
    state = State((("light", "off"), ("payoff", None), ("turn", "me")))
    observation = Observation(PythonRule("('light',)"), PythonRule("[({'light': 'on'}, 0.5), ({'light': 'off'}, 0.5)]"))

    domain = (
        DomainBuilder()
        .with_name("light")
        .with_initial_state(state)
        .with_problem(Problem(()))
        .with_transitions(TransitionModel(()))
        .with_players(Players(("me",), "turn", ("payoff",)))
        .with_observation(observation)
        .build()
    )

    assert domain.observation == observation


def test_build_rejects_missing_parts() -> None:
    builder = DomainBuilder().with_name("light").with_problem(Problem(()))

    with pytest.raises(ValueError, match="initial state, transitions, players"):
        builder.build()
