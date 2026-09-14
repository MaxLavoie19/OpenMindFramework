import numpy as np
import pytest

from openmind.training.model.non_inferiority import NonInferiority
from openmind.training.service.non_inferiority_test import NonInferiorityTest


def compare(differences: list[float], margin: float = 0.005) -> NonInferiority:
    return NonInferiorityTest().test(np.array(differences), margin, 0.95, 2000, np.random.default_rng(1))


def test_small_differences_around_zero_are_no_worse() -> None:
    result = compare([0.0, -0.01, 0.01, 0.0] * 50)

    assert (result.positions, result.regret_difference) == (200, 0.0)
    assert 0.0 < result.upper_bound < 0.005
    assert result.holds


def test_a_rise_in_regret_beyond_the_margin_is_worse() -> None:
    result = compare([0.0, 0.02] * 100)

    assert result.regret_difference == pytest.approx(0.01)
    assert result.upper_bound >= result.regret_difference
    assert not result.holds


def test_equal_differences_have_their_value_as_bound() -> None:
    assert (compare([0.0] * 10).upper_bound, compare([0.0] * 10).holds) == (0.0, True)
    assert (compare([0.01] * 10).upper_bound, compare([0.01] * 10).holds) == (0.01, False)
