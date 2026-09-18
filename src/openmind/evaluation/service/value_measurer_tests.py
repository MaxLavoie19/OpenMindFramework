import pytest

from openmind.evaluation.service.choice_measurer import ChoiceMeasurer
from openmind.evaluation.service.exact_search_tests import Declare, new_exact_search, trust
from openmind.evaluation.service.value_measurer import ValueMeasurer
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class Fixed:
    """Values every position the same, or knows nothing about any."""

    def __init__(self, values: tuple[float, ...] | None) -> None:
        self._values = values

    def values(self, state: State) -> tuple[float, ...] | None:
        return self._values


def new_measurer() -> ValueMeasurer:
    return ValueMeasurer(
        StateReader(), ChoiceMeasurer(StateTextMapper(), ActionTextMapper())
    )


def test_a_position_gives_the_value_error_and_the_choice_one_step_ahead(declared: Declare) -> None:
    rbs = trust(declared)
    values = new_exact_search().action_values(rbs, rbs.start())

    ((error, optimal, regret),) = new_measurer().measure(rbs, Fixed((0.9, 0.1)), [(rbs.start(), values)], 0.0)

    # A's best is safe, worth 0.5. One step ahead, safe ends the game at 0.5 for A, and risky leads to a position
    # valued 0.9 for A: risky is chosen, and it loses 0.5.
    assert (error, optimal, regret) == (pytest.approx(0.4), 0.0, 0.5)


def test_an_unvalued_outcome_gets_the_mean_of_the_other_actions_values(declared: Declare) -> None:
    rbs = trust(declared)
    values = new_exact_search().action_values(rbs, rbs.start())

    ((error, optimal, regret),) = new_measurer().measure(rbs, Fixed(None), [(rbs.start(), values)], 0.0)

    # Risky gets safe's 0.5, so both are top-valued: half the choices are optimal, and they lose 0.25 on average.
    assert (error, optimal, regret) == (None, 0.5, 0.25)
