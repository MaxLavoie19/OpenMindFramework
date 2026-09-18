import random

import pytest

from openmind.evaluation.service.exact_search_tests import Declare, trust
from openmind.evaluation.service.reference_search import ReferenceSearch
from openmind.world.model.action import Action
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")


def test_positions_are_distinct_positions_with_a_legal_action_from_random_games(declared: Declare) -> None:
    rbs = trust(declared)

    positions = ReferenceSearch().positions(rbs, 2, random.Random(1))

    assert set(positions) == {rbs.start(), State((("payoff(A)", None), ("payoff(B)", None), ("turn", "B")))}


def test_action_values_come_from_a_long_unguided_search(declared: Declare) -> None:
    rbs = trust(declared)

    values = dict(ReferenceSearch().action_values(rbs, rbs.start(), 200, 1))

    assert set(values) == {Action("safe", ()), Action("risky", ())}
    assert values[Action("safe", ())] == 0.5
    assert values[Action("risky", ())] < 0.5
