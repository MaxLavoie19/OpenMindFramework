import pytest

from openmind.timing.model.deadline import Deadline


class Ticking:
    """A time source a test moves by hand, so nothing here waits on a real clock."""

    def __init__(self, at: float = 0.0) -> None:
        self.at = at

    def now(self) -> float:
        return self.at


def test_a_deadline_is_that_many_seconds_from_now_on_its_own_source():
    source = Ticking(100.0)

    assert Deadline.after(5.0, source).at == pytest.approx(105.0)


def test_what_is_left_falls_as_the_source_moves_on():
    source = Ticking(0.0)
    deadline = Deadline.after(10.0, source)

    source.at = 4.0

    assert deadline.remaining() == pytest.approx(6.0)
    assert not deadline.passed()


def test_a_deadline_reached_exactly_has_passed():
    """Whatever must stop by then has to stop at the moment, not after it: a search asking `passed` on the tick
    and being told no would run one more node than it was given."""
    source = Ticking(0.0)
    deadline = Deadline.after(10.0, source)

    source.at = 10.0

    assert deadline.passed()
    assert deadline.remaining() == pytest.approx(0.0)


def test_what_is_left_goes_below_zero_once_it_has_passed():
    """By how much something overran is worth knowing, and a deadline clamped at zero could not say."""
    source = Ticking(0.0)
    deadline = Deadline.after(1.0, source)

    source.at = 4.0

    assert deadline.remaining() == pytest.approx(-3.0)


def test_it_carries_its_source_so_whatever_must_stop_needs_only_the_deadline():
    """A search handed a moment and no source could not tell whether the moment had come."""
    source = Ticking(0.0)

    assert Deadline.after(1.0, source).source is source
