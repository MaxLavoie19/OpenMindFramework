import numpy as np
import pytest

from openmind.rbs.model.row_arrays import RowArrays
from openmind.rbs.service.advantage_contrast import AdvantageContrast


def test_each_state_with_both_kinds_of_actions_gives_the_difference_of_their_weighted_advantages() -> None:
    arrays = RowArrays(
        states=np.array([0, 0, 0, 1, 1, 2]),
        visits=np.array([10.0, 30.0, 20.0, 10.0, 10.0, 5.0]),
        advantages=np.array([0.0, -0.4, -1.0, 0.0, -0.2, 0.0]),
        payoffs=np.zeros(6),
    )
    matching = np.array([True, True, False, False, True, True])

    differences = AdvantageContrast().differences(arrays, np.ones(6, dtype=bool), matching)

    # state 0: (10 * 0 + 30 * -0.4) / 40 - (-1.0) = 0.7; state 1: -0.2 - 0 = -0.2; state 2 has no other action
    assert differences.tolist() == pytest.approx([0.7, -0.2])


def test_the_scope_leaves_out_the_actions_outside_it() -> None:
    arrays = RowArrays(
        states=np.array([0, 0, 0]), visits=np.ones(3), advantages=np.array([0.0, -0.5, -1.0]), payoffs=np.zeros(3)
    )

    differences = AdvantageContrast().differences(
        arrays, np.array([True, True, False]), np.array([True, False, False])
    )

    assert differences.tolist() == pytest.approx([0.5])
