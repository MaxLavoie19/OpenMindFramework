import math

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.evaluation.service.choice_measurer import ChoiceMeasurer
from openmind.evaluation.service.evaluator_tests import first_mover_decides
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.model.action import Action

pytestmark = pytest.mark.log_level("INFO")

WIN, TIE, LOSE = Action("win", ()), Action("tie", ()), Action("lose", ())


def new_measurer() -> ChoiceMeasurer:
    return ChoiceMeasurer(StateTextMapper(), ActionTextMapper())


def test_a_choice_is_measured_against_the_action_values() -> None:
    domain = first_mover_decides()
    builder = AgentBuilder().with_iterations(30).with_exploration(math.sqrt(2)).with_seed(1)
    values = ((WIN, 1.0), (LOSE, 0.0), (TIE, 0.5))

    (measure,) = new_measurer().measure(domain, builder, ((domain.initial_state, values),), 0.0, "Agreement", 30)

    assert (measure.optimal, measure.regret) == (True, 0.0)
    assert 0.0 < measure.optimal_visit_share < 1.0
    assert measure.seconds >= 0.0


def test_optimal_actions_are_the_best_or_within_the_tolerance_of_it() -> None:
    values = ((WIN, 1.0), (TIE, 0.97), (LOSE, 0.5))

    assert new_measurer().optimal(values, 0.0) == frozenset({WIN})
    assert new_measurer().optimal(values, 0.05) == frozenset({WIN, TIE})
