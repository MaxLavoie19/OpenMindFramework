import pytest

from openmind.inference.model.example import Example
from openmind.inference.service.chance_fitter import ChanceFitter
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant


def case(holds: bool) -> Example:
    return Example((Literal("read", (Constant("something"),)),), holds)


def test_something_that_always_held_is_likely_and_never_certain() -> None:
    found = ChanceFitter().counted(10, 10)

    assert found.value > 0.8 and found.value < 1.0


def test_something_that_never_held_is_unlikely_and_never_impossible() -> None:
    found = ChanceFitter().counted(0, 10)

    assert found.value < 0.2 and found.value > 0.0


def test_nothing_counted_at_all_leaves_it_an_even_thing() -> None:
    assert ChanceFitter().counted(0, 0).value == 0.5


def test_the_same_share_seen_more_often_comes_back_with_a_narrower_spread() -> None:
    fitter = ChanceFitter()

    few = fitter.counted(5, 6)
    many = fitter.counted(5_000, 6_000)

    assert many.spread < few.spread


def test_a_share_seen_only_a_few_times_is_held_back_toward_an_even_thing() -> None:
    """Five out of six and five thousand out of six thousand are the same share and are not the same evidence. The
    scant one is pulled toward the middle, because six cases are as much use for telling 0.83 from 0.75 as they
    are for telling it from 0.9 — and the plentiful one is left where the counting put it."""
    fitter = ChanceFitter()

    few = fitter.counted(5, 6)
    many = fitter.counted(5_000, 6_000)

    assert few.value < many.value
    assert abs(many.value - 5_000 / 6_000) < 0.001


def test_more_counting_makes_one_estimate_preferable_to_another() -> None:
    fitter = ChanceFitter()

    assert fitter.surer(fitter.counted(5_000, 6_000), fitter.counted(5, 6))


def test_a_rule_is_fitted_on_the_cases_it_speaks_about_and_not_on_the_rest() -> None:
    clause = Clause((Literal("holds", ()),))
    examples = [case(True), case(True), case(False), case(True)]

    found = ChanceFitter().fit(clause, examples)

    assert found.derivations == 4 and 0.6 < found.value < 0.9


def test_counting_more_than_was_seen_is_refused() -> None:
    with pytest.raises(ValueError):
        ChanceFitter().counted(5, 3)


def test_a_fitted_chance_is_counted_rather_than_estimated() -> None:
    assert ChanceFitter().counted(3, 4).exact
