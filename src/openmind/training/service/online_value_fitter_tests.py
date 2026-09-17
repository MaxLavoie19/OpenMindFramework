import numpy as np
import pytest

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rule.model.python_rule import PythonRule
from openmind.training.service.online_value_fitter import OnlineValueFitter

PIECES, MOVES = PythonRule("pieces"), PythonRule("moves")


def base() -> ValueBase:
    return ValueBase("chess", 0.0, 0.0, 1.0, (ValueRule(PIECES, 0.5), ValueRule(MOVES, 0.5)))


def test_a_weight_whose_term_reads_higher_where_the_targets_are_higher_grows() -> None:
    terms = {PIECES: np.array([1.0, -1.0, 1.0, -1.0]), MOVES: np.array([0.0, 0.0, 0.0, 0.0])}

    stepped, largest = OnlineValueFitter().step(base(), terms, np.array([1.0, 0.0, 1.0, 0.0]), 0.1, 0.0)  # type: ignore[misc]

    weights = {rule.term: rule.weight for rule in stepped.rules}
    assert weights[PIECES] > 0.5 and weights[MOVES] == 0.5
    assert largest == pytest.approx(weights[PIECES] - 0.5)


def test_the_price_shrinks_a_weight_and_one_landing_on_zero_drops_its_rule() -> None:
    small = ValueBase("chess", 0.0, 0.0, 1.0, (ValueRule(PIECES, 0.001), ValueRule(MOVES, 0.5)))
    terms = {PIECES: np.zeros(2), MOVES: np.zeros(2)}

    stepped, _ = OnlineValueFitter().step(small, terms, np.array([0.5, 0.5]), 0.1, 0.1)  # type: ignore[misc]

    assert [rule.term for rule in stepped.rules] == [MOVES]
    assert stepped.rules[0].weight == pytest.approx(0.49)


def test_a_blank_reading_adds_nothing_and_a_term_that_couldn_t_be_read_or_no_row_gives_no_step() -> None:
    fitter = OnlineValueFitter()
    blank = {PIECES: np.array([np.nan, np.nan]), MOVES: np.array([np.nan, np.nan])}

    stepped, _ = fitter.step(base(), blank, np.array([1.0, 1.0]), 0.1, 0.0)  # type: ignore[misc]

    assert [rule.weight for rule in stepped.rules] == [0.5, 0.5] and stepped.bias > 0.0
    assert fitter.step(base(), {PIECES: np.zeros(2), MOVES: None}, np.array([1.0, 1.0]), 0.1, 0.0) is None
    assert fitter.step(base(), {PIECES: np.zeros(0), MOVES: np.zeros(0)}, np.zeros(0), 0.1, 0.0) is None
