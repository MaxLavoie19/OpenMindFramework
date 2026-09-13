import pytest

from openmind.agent.builder.domain_builder import DomainBuilder
from openmind.agent.model.domain import Domain
from openmind.csp.model.problem import Problem
from openmind.predictor.model.transition_model import TransitionModel
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


def test_build_rejects_missing_parts() -> None:
    builder = DomainBuilder().with_name("light").with_problem(Problem(()))

    with pytest.raises(ValueError, match="initial state, transitions, players"):
        builder.build()
