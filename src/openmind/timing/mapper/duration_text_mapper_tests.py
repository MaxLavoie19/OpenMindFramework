from openmind.timing.mapper.duration_text_mapper import DurationTextMapper


def said(seconds):
    return DurationTextMapper().to_text(seconds)


def test_a_short_wait_is_said_in_seconds():
    assert said(0.2) == "0.2 seconds"
    assert said(1) == "1 second"
    assert said(45) == "45 seconds"


def test_a_longer_one_is_said_in_minutes_and_seconds():
    assert said(60) == "1 minute"
    assert said(119) == "1 minute 59 seconds"
    assert said(587) == "9 minutes 47 seconds"


def test_hours_and_days_are_said_with_the_unit_below_them():
    """Nobody reads 4916 seconds as an hour and twenty minutes without doing the arithmetic."""
    assert said(3600) == "1 hour"
    assert said(4916) == "1 hour 22 minutes"
    assert said(90000) == "1 day 1 hour"


def test_what_rounds_up_to_a_whole_unit_is_said_as_one():
    assert said(3599.8) == "1 hour"
    assert said(59.6) == "1 minute"


def test_time_going_backwards_is_said_so_rather_than_hidden():
    assert said(-90) == "minus 1 minute 30 seconds"
