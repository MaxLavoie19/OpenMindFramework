import math
import pickle
from dataclasses import replace

import pytest

from openmind.rbs.factory.rbs_factory import create_rule_valuer
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rbs.service.consequence_library_tests import position, strip_domain
from openmind.rule.model.python_rule import PythonRule

BASE = ValueBase(
    "strip", 0.0, 0.0, 1.0, (ValueRule(PythonRule("wins(me)"), 2.0), ValueRule(PythonRule("turn == me"), -1.0))
)
#: X has a mark and could win next to it; O is to act.
X_THREATENS = position({1: "X"}, "O")


def logistic(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))


def test_each_player_is_valued_with_me_being_that_player() -> None:
    values = create_rule_valuer(BASE, strip_domain()).value(X_THREATENS)

    # X: wins(me) is 1 and it isn't X's turn; O: wins(me) is 0 and it is O's turn.
    assert values == pytest.approx((logistic(2.0), logistic(-1.0)))


def test_low_and_high_scale_the_logistic() -> None:
    base = replace(BASE, low=-1.0, high=3.0, rules=())

    assert create_rule_valuer(base, strip_domain()).value(position({}, "X")) == (1.0, 1.0)


def test_explain_gives_what_each_rule_adds_to_the_score() -> None:
    assert create_rule_valuer(BASE, strip_domain()).explain(X_THREATENS, "X") == ((BASE.rules[0], 2.0), (BASE.rules[1], 0.0))


def test_a_blank_term_adds_nothing() -> None:
    blank = ValueRule(PythonRule("1 if wins(me) else None"), 2.0)
    base = replace(BASE, rules=(blank, BASE.rules[1]))
    valuer = create_rule_valuer(base, strip_domain())

    # X: the term is 1; O can't win next, so the term is blank and only O's turn counts.
    assert valuer.value(X_THREATENS) == pytest.approx((logistic(2.0), logistic(-1.0)))
    assert valuer.explain(X_THREATENS, "O") == ((blank, 0.0), (BASE.rules[1], -1.0))


@pytest.mark.parametrize("source", ["lamp", "turn"])
def test_a_term_that_raises_or_gives_no_number_leaves_the_position_unvalued(source: str) -> None:
    base = replace(BASE, rules=(ValueRule(PythonRule(source), 1.0),))

    assert create_rule_valuer(base, strip_domain()).value(X_THREATENS) is None


def test_a_copy_sent_to_another_process_values_the_same() -> None:
    valuer = create_rule_valuer(BASE, strip_domain())
    values = valuer.value(X_THREATENS)

    assert pickle.loads(pickle.dumps(valuer)).value(X_THREATENS) == values
