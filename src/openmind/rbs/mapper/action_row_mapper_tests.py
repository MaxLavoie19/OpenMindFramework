import pytest

from openmind.mcts.model.action_sample import ActionSample
from openmind.rbs.mapper.action_row_mapper import ActionRowMapper
from openmind.rbs.model.action_row import ActionRow
from openmind.world.model.action import Action
from openmind.world.model.state import State

A, B = State((("turn", "X"), ("step", 1))), State((("turn", "X"), ("step", 2)))


def press(cell: int) -> Action:
    return Action("press", (("cell", cell),))


def test_samples_of_an_action_in_a_state_merge_and_advantages_compare_with_the_best_action() -> None:
    samples = [
        ActionSample(A, 0, press(1), 10, 1.0),
        ActionSample(A, 0, press(1), 30, 0.6),
        ActionSample(A, 0, press(2), 20, 0.2),
    ]

    rows = ActionRowMapper().to_rows(samples, min_visits=5)

    assert [(row.action, row.visits, row.mean_payoff, row.advantage) for row in rows] == [
        (press(1), 40, pytest.approx(0.7), 0.0),
        (press(2), 20, 0.2, pytest.approx(-0.5)),
    ]


def test_rows_under_min_visits_and_states_left_with_one_row_are_dropped() -> None:
    samples = [
        ActionSample(A, 0, press(1), 10, 1.0),
        ActionSample(A, 0, press(2), 2, 0.0),
        ActionSample(B, 0, press(1), 10, 0.5),
        ActionSample(B, 0, press(2), 10, 0.5),
    ]

    assert ActionRowMapper().to_rows(samples, min_visits=5) == (
        ActionRow(B, press(1), 10, 0.5, 0.0),
        ActionRow(B, press(2), 10, 0.5, 0.0),
    )


def test_arrays_give_rows_of_the_same_state_the_same_index() -> None:
    rows = (ActionRow(A, press(1), 10, 1.0, 0.0), ActionRow(B, press(1), 5, 0.5, 0.0), ActionRow(A, press(2), 20, 0.2, -0.8))

    arrays = ActionRowMapper().to_arrays(rows)

    assert (arrays.states.tolist(), arrays.visits.tolist(), arrays.advantages.tolist(), arrays.payoffs.tolist()) == (
        [0, 1, 0],
        [10.0, 5.0, 20.0],
        [0.0, 0.0, -0.8],
        [1.0, 0.5, 0.2],
    )
