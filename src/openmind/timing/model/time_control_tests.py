import pytest

from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl


def test_a_player_starts_with_the_base_time_and_the_increment() -> None:
    assert TimeControl(180.0, 2.0).clock() == Clock(180.0, 2.0, False)


def test_a_time_control_without_an_increment_adds_nothing() -> None:
    assert TimeControl(60.0).clock().after(10.0).remaining == 50.0


@pytest.mark.parametrize(("base", "increment"), [(0.0, 2.0), (-60.0, 0.0), (60.0, -1.0)])
def test_a_base_not_above_zero_or_a_negative_increment_raises(base: float, increment: float) -> None:
    with pytest.raises(ValueError):
        TimeControl(base, increment)
