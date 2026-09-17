import pytest

from openmind.timing.model.clock import Clock
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator

pytestmark = pytest.mark.log_level("INFO")


def test_a_step_gets_the_time_left_over_the_steps_expected_plus_the_increment() -> None:
    assert PlainTimeBudgetEstimator(30).budget(Clock(180.0, 2.0), 0) == 8.0


def test_without_an_increment_a_step_gets_the_time_left_over_the_steps_expected() -> None:
    assert PlainTimeBudgetEstimator(30).budget(Clock(60.0), 0) == 2.0


def test_a_step_never_gets_more_than_the_time_left() -> None:
    assert PlainTimeBudgetEstimator(30).budget(Clock(1.0, 2.0), 0) == 1.0


def test_the_steps_played_change_nothing_for_the_plain_rule() -> None:
    estimator = PlainTimeBudgetEstimator(30)

    assert estimator.budget(Clock(180.0, 2.0), 0) == estimator.budget(Clock(180.0, 2.0), 75)


@pytest.mark.parametrize(
    ("expected_steps", "clock", "steps_played", "message"),
    [
        (30, Clock(0.0, 2.0, True), 0, "ran out"),
        (30, Clock(60.0), -1, "negative"),
        (0, Clock(60.0), 0, "At least 1 step"),
    ],
)
def test_a_flagged_clock_negative_steps_or_no_step_expected_raise(
    expected_steps: int, clock: Clock, steps_played: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PlainTimeBudgetEstimator(expected_steps).budget(clock, steps_played)


def test_logs_the_budget_and_how_it_was_reached(caplog: pytest.LogCaptureFixture) -> None:
    estimator = PlainTimeBudgetEstimator(30)

    estimator.budget(Clock(61.0, 2.0), 3)
    estimator.budget(Clock(1.0, 2.0), 40)

    assert caplog.messages == [
        "A step may take 4.033 seconds: 61.0 seconds left above a 0 second reserve over 30 steps expected, plus a 2 second increment",
        "A step may take 1.000 seconds: 1.0 seconds left above a 0 second reserve over 30 steps expected, plus a 2 second "
        "increment, limited to the time left",
    ]


def test_the_reserve_is_kept_out_of_every_budget_and_a_clock_at_or_below_it_gets_none(caplog: pytest.LogCaptureFixture) -> None:
    estimator = PlainTimeBudgetEstimator(10, 15.0)

    assert estimator.budget(Clock(115.0, 1.0), 0) == 11.0
    assert estimator.budget(Clock(15.5, 2.0), 0) == 0.5
    assert estimator.budget(Clock(15.0, 2.0), 0) == 0.0
    assert estimator.budget(Clock(3.0), 0) == 0.0
    assert "A step may take 0 seconds: 3.0 seconds left, at or below the 15 second reserve" in caplog.messages
    with pytest.raises(ValueError, match="reserve can't be negative"):
        PlainTimeBudgetEstimator(10, -1.0)
