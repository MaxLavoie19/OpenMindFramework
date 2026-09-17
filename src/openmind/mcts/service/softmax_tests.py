import math

import pytest

from openmind.mcts.service.softmax import softmax


def test_shares_follow_the_values_more_closely_the_lower_the_temperature() -> None:
    warm, cold = softmax((0.6, 0.5), 1.0), softmax((0.6, 0.5), 0.01)

    assert math.isclose(sum(warm), 1.0) and math.isclose(sum(cold), 1.0)
    assert 0.5 < warm[0] < 0.6
    assert cold[0] > 0.9999


def test_an_unknown_value_takes_the_mean_of_the_known_ones_and_with_none_known_every_share_is_equal() -> None:
    assert softmax((0.2, None, 0.4), 0.1)[1] == pytest.approx(softmax((0.2, 0.3, 0.4), 0.1)[1])
    assert softmax((None, None), 0.1) == (0.5, 0.5)


def test_a_temperature_of_zero_raises() -> None:
    with pytest.raises(ValueError, match="above 0"):
        softmax((1.0,), 0.0)
