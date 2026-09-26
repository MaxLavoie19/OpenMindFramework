import pytest

from openmind.timing.model.time_control import TimeControl


def test_a_player_starts_with_the_base_and_gains_the_increment_each_step():
    """Three minutes and two seconds a move, which is what a time control is for saying."""
    clock = TimeControl(180, 2).clock()

    assert clock.remaining == pytest.approx(180.0)
    assert clock.increment == pytest.approx(2.0)
    assert not clock.flagged


def test_a_control_with_no_increment_gives_a_clock_that_only_ever_runs_down():
    assert TimeControl(60).clock().after(10.0).remaining == pytest.approx(50.0)


def test_a_base_of_nothing_is_refused_rather_than_giving_a_game_nobody_can_move_in():
    """A clock starting at zero is flagged before the first move, so the game is lost before it is played —
    which is a mistake worth catching where it is made rather than where it shows."""
    with pytest.raises(ValueError, match="above zero"):
        TimeControl(0)


def test_a_negative_base_is_refused():
    with pytest.raises(ValueError, match="above zero"):
        TimeControl(-1)


def test_a_negative_increment_is_refused():
    """An increment taking time away is not an increment, and a clock would gain speed by being played slowly."""
    with pytest.raises(ValueError, match="can't be negative"):
        TimeControl(60, -1)


def test_each_player_gets_a_clock_of_its_own():
    """Two players sharing one clock would spend each other's time; a control hands out a fresh one each time."""
    control = TimeControl(60, 1)

    first, second = control.clock(), control.clock()

    assert first == second
    assert first.after(10.0) != second
