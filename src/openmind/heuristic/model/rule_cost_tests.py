import pytest

from openmind.heuristic.model.rule_cost import RuleCost

pytestmark = pytest.mark.log_level("INFO")


def test_a_rule_nobody_has_timed_is_priced_at_nothing() -> None:
    """So that it is tried once and measured, rather than passed over for ever on the strength of never
    having been read. That is the exploration a measured cost needs."""
    assert RuleCost().priced() == 0.0


def test_one_reading_gives_a_mean_and_no_spread() -> None:
    """A spread over one reading is not a small spread, it is no spread at all."""
    cost = RuleCost().with_reading(0.4)

    assert cost.seconds == pytest.approx(0.4)
    assert cost.spread == 0.0
    assert cost.read == 1


def test_what_a_rule_costs_is_the_mean_of_what_it_has_cost() -> None:
    cost = RuleCost()
    for seconds in (0.1, 0.2, 0.3):
        cost = cost.with_reading(seconds)

    assert cost.seconds == pytest.approx(0.2)
    assert cost.read == 3


def test_a_rule_that_varies_is_priced_above_a_rule_that_does_not() -> None:
    """**Why a mean is not enough.** A rule that walks the threats in a position is cheap where nothing is
    threatened and dear where everything is. What a caller needs to know is whether the dear reading fits in
    the time left, not whether the average one does."""
    steady, jumpy = RuleCost(), RuleCost()
    for seconds in (0.2, 0.2, 0.2, 0.2):
        steady = steady.with_reading(seconds)
    for seconds in (0.05, 0.35, 0.05, 0.35):
        jumpy = jumpy.with_reading(seconds)

    assert steady.seconds == pytest.approx(jumpy.seconds), "the same on average"
    assert jumpy.priced() > steady.priced(), "and not the same to rely on"


def test_more_caution_prices_a_varying_rule_higher_and_a_steady_one_the_same() -> None:
    """The caution buys certainty where there is variance to be uncertain about, and nothing where there is
    none — the same reading the search already makes of what a node costs."""
    steady, jumpy = RuleCost(), RuleCost()
    for seconds in (0.2, 0.2, 0.2):
        steady = steady.with_reading(seconds)
    for seconds in (0.1, 0.2, 0.3):
        jumpy = jumpy.with_reading(seconds)

    assert steady.priced(6.0) == pytest.approx(steady.priced(1.0))
    assert jumpy.priced(6.0) > jumpy.priced(1.0)


def test_the_mean_and_spread_are_what_the_readings_say() -> None:
    """Welford, so nothing has to keep what it has seen. Costs of 1, 2 and 3 have a mean of 2 and a sample
    spread of 1."""
    cost = RuleCost()
    for seconds in (1.0, 2.0, 3.0):
        cost = cost.with_reading(seconds)

    assert cost.seconds == pytest.approx(2.0)
    assert cost.spread == pytest.approx(1.0)


def test_a_rule_read_rarely_is_hoped_cheaper_than_its_mean() -> None:
    """**Greedy starves what it declines.** A rule priced high is never taken, so it is never timed again, so
    one unlucky reading condemns it for ever. The discount is a standard error — the uncertainty itself, not a
    number anybody chose."""
    cost = RuleCost()
    for seconds in (0.1, 0.5):
        cost = cost.with_reading(seconds)

    assert cost.hoped() < cost.seconds


def test_the_hope_fades_as_the_readings_pile_up() -> None:
    """A rule read often has a cost worth believing, so it is given no benefit of any doubt."""
    few, many = RuleCost(), RuleCost()
    for seconds in (0.1, 0.5):
        few = few.with_reading(seconds)
    for _ in range(40):
        for seconds in (0.1, 0.5):
            many = many.with_reading(seconds)

    assert many.seconds == pytest.approx(few.seconds, abs=0.01), "the same cost on average"
    assert many.seconds - many.hoped() < few.seconds - few.hoped(), "and far less doubt about it"


def test_a_rule_whose_cost_never_varies_is_hoped_at_exactly_its_cost() -> None:
    """Exploration answers doubt, and there is none here. A rule that has cost the same every time is not
    being starved by being declined — it is being declined correctly."""
    cost = RuleCost()
    for _ in range(5):
        cost = cost.with_reading(0.3)

    assert cost.hoped() == pytest.approx(0.3)


def test_hope_and_caution_pull_opposite_ways_on_the_same_rule() -> None:
    """Which is the point, not a contradiction. What to try is decided optimistically and what fits is decided
    pessimistically: a budget overrun is a real cost where a missed rule is only an opportunity."""
    cost = RuleCost()
    for seconds in (0.1, 0.5):
        cost = cost.with_reading(seconds)

    assert cost.hoped() < cost.seconds < cost.priced()


def test_a_rule_read_once_is_hoped_free_so_it_gets_a_second_reading() -> None:
    """One reading is a mean with no spread, so there is nothing to be uncertain by. It is treated as unknown
    rather than as measured."""
    assert RuleCost().with_reading(9.0).hoped() == 0.0
