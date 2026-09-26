import pytest

from openmind.inference.service.statistics import Statistics


def test_nothing_is_certain_from_nothing():
    """**A refusal, not a guess about the world.** Counting alone says a rule seen once and held once always
    holds, and says it with the same face as a rule seen ten thousand times. It also reaches nought and one —
    and a rule believed impossible is never tried again, so it is never corrected."""
    held = Statistics()

    assert held.share(0, 0) == 0.5
    assert 0.0 < held.share(0, 10) < held.share(5, 10) < held.share(10, 10) < 1.0


def test_the_spread_is_what_tells_six_from_six_thousand():
    """Five of six and five thousand of six thousand are both 0.83 and are not the same claim. A middle without
    a spread cannot tell them apart, which is why nothing here returns one alone."""
    held = Statistics()

    assert held.spread(5, 6) > held.spread(5000, 6000)
    assert abs(held.share(5, 6) - held.share(5000, 6000)) < 0.15


def test_what_something_claims_of_itself_is_outvoted_by_what_is_seen():
    """A mechanism saying how accurate it is has said something worth keeping until enough has been seen to say
    otherwise. Weighted at two cases, the third observation outvotes it — which is the right order of trust for
    a claim nobody has checked."""
    held = Statistics()

    assert held.leaning(0, 0, toward=0.9, weight=2.0) == pytest.approx(0.9)
    assert held.leaning(0, 10, toward=0.9, weight=2.0) < 0.2
    assert held.leaning(10, 10, toward=0.1, weight=2.0) > 0.8


def test_counting_more_than_was_seen_is_refused_rather_than_shrugged_at():
    """It is a fault in the caller, and quietly returning something over one would carry it into everything
    downstream."""
    with pytest.raises(ValueError):
        Statistics().share(11, 10)


def test_how_far_numbers_spread_about_their_middle():
    """The population's spread and not the sample's: these are what was seen rather than a draw from something
    larger, so a mechanism's errors are its errors."""
    held = Statistics()

    assert held.middle([]) == 0.0
    assert held.apart([]) == 0.0
    assert held.apart([3.0, 3.0, 3.0]) == 0.0
    assert held.apart([2.0, 4.0]) == pytest.approx(1.0)


def test_what_is_assumed_before_counting_is_the_callers():
    """How much to assume is a question about what is being counted, not about counting."""
    assert Statistics(before_holding=9.0, before_not=1.0).share(0, 0) == 0.9
