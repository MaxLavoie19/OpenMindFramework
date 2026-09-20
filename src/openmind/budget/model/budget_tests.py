from openmind.budget.model.budget import Budget
from openmind.timing.model.clock import Clock


def test_carving_a_child_s_seconds_leaves_the_parent_that_much_less() -> None:
    child, left = Budget(10.0).carve(3.0)

    assert (child.seconds, left.seconds) == (3.0, 7.0)


def test_carving_more_than_there_is_gives_what_there_is_and_leaves_nothing() -> None:
    child, left = Budget(2.0).carve(5.0)

    assert (child.seconds, left.seconds) == (2.0, 0.0)


def test_a_carved_budget_runs_on_the_clock_it_was_carved_from() -> None:
    clock = Clock(60.0)

    child, left = Budget(10.0, clock).carve(3.0)

    assert child.of_clock is clock and left.of_clock is clock


def test_a_budget_running_alongside_takes_nothing_from_the_parent_s() -> None:
    parent = Budget(10.0)

    child = parent.alongside(3.0)

    assert (child.seconds, parent.seconds) == (3.0, 10.0)
