import math

import numpy as np
import pytest

from openmind.inference.model.choice import Choice
from openmind.inference.service.choice_fitter import ChoiceFitter


def _chose(taken: int, *candidates: tuple[float, ...]) -> Choice:
    return Choice(np.array(candidates, dtype=float), taken)


def test_what_was_always_preferred_gets_a_weight_and_what_was_not_gets_none() -> None:
    """The whole of it: nobody said what anything was worth, and the preference is read off the comparison."""
    # Two terms. The first is what they went for every time; the second has nothing to do with the choice.
    choices = [_chose(0, (1.0, 1.0), (0.0, 1.0)), _chose(0, (1.0, 0.0), (0.0, 0.0)), _chose(1, (0.0, 1.0), (1.0, 1.0))]

    fitted = ChoiceFitter().fit(choices, price=0.0, max_steps=500, tolerance=1e-6)

    assert fitted.weights[0] > 0.0, "the term they went for is worth something"
    assert abs(fitted.weights[1]) < abs(fitted.weights[0]), "the term that said nothing is worth less"


def test_a_preference_has_no_level_to_fit() -> None:
    """A constant added to every candidate of a decision cancels in the choosing, so a bias is a direction
    the likelihood is flat along and fitting one would be fitting noise."""
    fitted = ChoiceFitter().fit([_chose(0, (1.0,), (0.0,))], price=0.0, max_steps=100, tolerance=1e-6)

    assert fitted.bias == 0.0


def test_the_price_shrinks_a_term_that_earns_little_to_nothing() -> None:
    """The same handle the sparse fitter offers, and for the same reason: a weight that does not pay its
    price lands on zero rather than merely near it."""
    choices = [_chose(0, (1.0, 0.3), (0.0, 0.0)), _chose(0, (1.0, 0.0), (0.0, 0.3))]
    fitter = ChoiceFitter()

    free = fitter.fit(choices, price=0.0, max_steps=500, tolerance=1e-8)
    priced = fitter.fit(choices, price=0.2, max_steps=500, tolerance=1e-8)

    assert free.weights[1] != 0.0
    assert priced.weights[1] == 0.0, "exactly nought, not merely small"


def test_a_term_that_is_dear_to_read_is_dear_to_keep() -> None:
    """Costs per column, so what a term costs to read bears on whether it survives."""
    choices = [_chose(0, (1.0, 1.0), (0.0, 0.0)), _chose(0, (1.0, 1.0), (0.0, 0.0))]
    fitter = ChoiceFitter()

    even = fitter.fit(choices, price=0.1, max_steps=500, tolerance=1e-8)
    dearer = fitter.fit(choices, price=0.1, max_steps=500, tolerance=1e-8, costs=np.array([1.0, 20.0]))

    assert abs(dearer.weights[1]) < abs(even.weights[1]), "the dear one is shrunk harder"


def test_knowing_nothing_is_what_the_loss_is_read_against() -> None:
    """A loss on its own says nothing. Ten candidates and a loss of 2.30 is a fit that learned exactly
    nothing; the same loss where two were on offer is worse than a coin."""
    choices = [_chose(0, *[(float(at),) for at in range(10)])]
    fitter = ChoiceFitter()

    assert fitter.knowing_nothing(choices) == pytest.approx(math.log(10))
    assert fitter.loss(choices, (0.0,)) == pytest.approx(math.log(10)), "no weights is knowing nothing"


def test_a_fit_that_learned_is_less_surprised_than_one_that_did_not() -> None:
    choices = [_chose(0, (1.0,), (0.0,)), _chose(0, (1.0,), (0.0,)), _chose(0, (1.0,), (0.0,))]
    fitter = ChoiceFitter()

    fitted = fitter.fit(choices, price=0.0, max_steps=500, tolerance=1e-8)

    assert fitter.loss(choices, fitted.weights) < fitter.knowing_nothing(choices)


def test_a_decision_offering_one_thing_is_not_a_decision() -> None:
    """Nothing was chosen between, so there is nothing in it to learn from."""
    fitter = ChoiceFitter()
    alone = [_chose(0, (1.0,))]

    assert fitter.loss(alone, (1.0,)) == 0.0
    with pytest.raises(ValueError):
        fitter.fit(alone, price=0.0, max_steps=10, tolerance=1e-6)


def test_decisions_read_in_different_terms_cannot_be_fitted_together() -> None:
    choices = [_chose(0, (1.0, 0.0), (0.0, 1.0)), _chose(0, (1.0,), (0.0,))]

    with pytest.raises(ValueError):
        ChoiceFitter().fit(choices, price=0.0, max_steps=10, tolerance=1e-6)


def test_how_many_were_on_offer_is_part_of_the_evidence() -> None:
    """A move taken from three says little and the same move taken from forty says a great deal. A fit shown
    only what was played could not tell them apart."""
    fitter = ChoiceFitter()
    few = [_chose(0, (1.0,), (0.0,))]
    many = [_chose(0, (1.0,), *[(0.0,)] * 39)]

    assert fitter.knowing_nothing(many) > fitter.knowing_nothing(few)
    assert fitter.fit(many, price=0.0, max_steps=500, tolerance=1e-8).weights[0] > 0.0
