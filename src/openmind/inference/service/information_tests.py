from math import isclose

from openmind.inference.service.information import Information


def test_nothing_is_learned_by_being_told_which_when_there_is_only_one():
    """Surprise is the whole of what a bit measures. Where everything is the same thing there was never a
    question, so being told the answer says nothing."""
    assert Information().entropy({"a": 40}) == 0.0
    assert Information().entropy({}) == 0.0


def test_the_most_is_said_where_nothing_could_be_guessed():
    """Equally likely is the case with nothing to go on, and it is where a symbol has most work to do. Four
    equal things take two bits; eight take three."""
    held = Information()

    assert isclose(held.entropy({one: 1 for one in range(4)}), 2.0)
    assert isclose(held.entropy({one: 1 for one in range(8)}), 3.0)
    assert held.entropy({"a": 99, "b": 1}) < held.entropy({"a": 50, "b": 50})


def test_what_knowing_saves_is_what_it_cost_before_less_what_it_costs_now():
    """Which is what a symbol *telling* us about a part means, and what the coupling learner measures."""
    held = Information()

    assert isclose(held.told({one: 1 for one in range(4)}, {"settled": 4}), 2.0)
    assert held.told({"a": 1, "b": 1}, {"a": 1, "b": 1}) == 0.0


def test_a_number_nobody_bounded_is_said_without_inventing_a_bound():
    """How many constraints a game needs is not something to assume. Rissanen's code says a whole number
    without one, and costs more for bigger ones — which is the pressure against a pile."""
    held = Information()

    assert held.counting(1) < held.counting(10) < held.counting(1000)
    assert held.counting(0) == held.counting(1), "nothing and one thing are both said at once"


def test_saying_which_of_a_pool_gets_dearer_more_slowly_the_more_are_taken():
    """The property that makes a code a code rather than a penalty: the marginal cost of a further condition
    falls, because there are fewer ways left to choose it. A made-up quadratic does the opposite."""
    held = Information()
    marginal = [held.choosing(200, one + 1) - held.choosing(200, one) for one in range(1, 8)]

    assert marginal == sorted(marginal, reverse=True)


def test_pointing_one_out_of_many_costs_more_than_one_out_of_few():
    """What makes a theory pay for itself: the fewer a set of rules leaves standing, the less it takes to say
    which of them happened."""
    held = Information()

    assert isclose(held.naming(14400), 13.81, abs_tol=0.01)
    assert isclose(held.naming(40), 5.32, abs_tol=0.01)
    assert held.naming(1) == 0.0


def test_an_unordered_set_is_not_charged_for_an_order_it_has_not_got():
    """K things could have been written down K! ways and they all mean the same thing."""
    assert isclose(Information().ordering(10), 21.79, abs_tol=0.01)
    assert Information().ordering(1) == 0.0
