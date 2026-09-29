import pytest

from openmind.inference.model.expression import Expression
from openmind.model.factory.model_factory import create_rule_price

pytestmark = pytest.mark.log_level("INFO")


def term(clauses: int) -> Expression:
    return Expression(template="VIEW", clauses=clauses, plies=0)


def test_a_longer_rule_costs_more_to_vouch_for() -> None:
    """The whole of what "more clauses are more expensive, so they exist but are rarer" means. A signal can buy
    a long rule; it has to have saved for it."""
    price = create_rule_price()
    prices = [price.priced(term(clauses), vocabulary=200) for clauses in range(1, 8)]

    assert prices == sorted(prices)
    assert prices[0] < prices[-1]


def test_each_further_condition_costs_less_than_the_one_before() -> None:
    """The reason the quadratic was struck. Under a real code the fifth condition is cheaper than the third,
    because there are fewer ways left to choose it; under `1 + k²` it is dearer, which prefers short rules —
    and a short rule is an over-general one. If this ever reverses, the price has stopped being a code."""
    price = create_rule_price()
    prices = [price.priced(term(clauses), vocabulary=200) for clauses in range(1, 12)]
    steps = [second - first for first, second in zip(prices, prices[1:])]

    assert steps == sorted(steps, reverse=True), "the marginal cost of a further condition falls"


def test_a_rule_drawn_from_a_bigger_pool_costs_more_to_name() -> None:
    """The vocabulary is a fact about the search that produced the candidates, not a knob. Naming one term of
    a thousand takes more saying than naming one of ten, and that shifts how many rules are worth buying."""
    price = create_rule_price()

    assert price.priced(term(3), vocabulary=1000) > price.priced(term(3), vocabulary=10)


def test_a_rule_that_reads_nothing_still_costs_something_to_say() -> None:
    """Nought conditions is a rule that there is a rule, and saying so is not free. A price of nought would be
    a rule any signal could buy for ever out of an empty purse."""
    assert create_rule_price().priced(term(0), vocabulary=200) > 0.0


def test_a_pool_smaller_than_the_rule_prices_rather_than_refusing() -> None:
    """A pool can shrink between the search that generated a term and the judging that prices it, and a term
    priced at infinity would be one no signal could ever vouch for on an accident of ordering."""
    priced = create_rule_price().priced(term(5), vocabulary=2)

    assert priced == pytest.approx(create_rule_price().priced(term(5), vocabulary=5))
