import pytest

from openmind.timing.model.clock import Clock


def test_a_step_takes_its_seconds_off_and_adds_the_increment():
    """The increment is what a player gets back for having moved, so it lands after the step and not before."""
    after = Clock(60.0, 2.0).after(10.0)

    assert after.remaining == pytest.approx(52.0)
    assert not after.flagged


def test_a_clock_reaching_zero_is_flagged_as_a_chess_flag_falls_at_zero():
    after = Clock(10.0).after(10.0)

    assert after.flagged
    assert after.remaining == 0.0


def test_a_flagged_clock_gains_no_increment_however_generous_it_was():
    """The game is lost on time; giving the increment back would hand a player time they never earned and,
    worse, unflag them."""
    after = Clock(1.0, 30.0).after(5.0)

    assert after.flagged
    assert after.remaining == pytest.approx(-4.0)


def test_what_is_left_below_zero_says_by_how_much_the_step_overran():
    """A step that overran by a tenth and one that overran by a minute are not the same thing to have done."""
    assert Clock(1.0).after(1.1).remaining == pytest.approx(-0.1)
    assert Clock(1.0).after(61.0).remaining == pytest.approx(-60.0)


def test_a_clock_that_ran_out_has_no_more_steps():
    with pytest.raises(ValueError, match="no more steps"):
        Clock(-1.0, flagged=True).after(1.0)


def test_a_step_cannot_take_negative_time():
    """Time going backwards would hand a player seconds by playing, which is not a thing a clock can mean."""
    with pytest.raises(ValueError, match="negative time"):
        Clock(60.0).after(-1.0)


def test_a_step_taking_no_time_still_earns_its_increment():
    """A move played instantly is a move played, and a clock that only rewarded slow moves would be a strange one."""
    assert Clock(60.0, 2.0).after(0.0).remaining == pytest.approx(62.0)
