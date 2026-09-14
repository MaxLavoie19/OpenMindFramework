import random

import pytest

from openmind.csp.factory.csp_factory import create_solver
from openmind.evaluation.service.exact_search_tests import trust
from openmind.evaluation.service.reference_search import ReferenceSearch
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.model.action import Action
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")


def test_positions_are_distinct_positions_with_a_legal_action_from_random_games() -> None:
    domain = trust()

    positions = ReferenceSearch(create_solver(), create_predictor()).positions(domain, 2, random.Random(1))

    assert set(positions) == {domain.initial_state, State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))}


def test_action_values_come_from_a_long_unguided_search() -> None:
    domain = trust()

    values = dict(ReferenceSearch(create_solver(), create_predictor()).action_values(domain, domain.initial_state, 200, 1))

    assert set(values) == {Action("safe", ()), Action("risky", ())}
    assert values[Action("safe", ())] == 0.5
    assert values[Action("risky", ())] < 0.5
