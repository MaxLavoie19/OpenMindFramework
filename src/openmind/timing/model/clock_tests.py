import pytest

from openmind.timing.model.clock import Clock


def test_a_step_takes_what_it_spent_and_gives_the_increment_back() -> None:
    clock = Clock(180.0, 2.0)

    after = clock.after(5.0)

    assert after == Clock(177.0, 2.0, False)


def test_a_step_reaching_zero_flags_the_clock_and_adds_nothing() -> None:
    assert Clock(3.0, 2.0).after(3.0) == Clock(0.0, 2.0, True)


def test_a_step_overrunning_the_time_left_flags_it_and_says_by_how_much() -> None:
    after = Clock(3.0, 2.0).after(4.5)

    assert after.flagged
    assert after.remaining == -1.5


def test_a_clock_that_ran_out_takes_no_more_steps() -> None:
    with pytest.raises(ValueError, match="no more steps"):
        Clock(0.0, 2.0, True).after(1.0)


def test_a_step_can_t_take_negative_time() -> None:
    with pytest.raises(ValueError, match="negative time"):
        Clock(60.0).after(-0.1)


def test_steps_add_up_as_a_game_goes_on() -> None:
    clock = Clock(60.0, 1.0)
    for spent in (10.0, 20.0, 15.0):
        clock = clock.after(spent)

    assert clock == Clock(18.0, 1.0, False)
