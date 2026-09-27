from openmind.inference.service.evaluable_predicates import EvaluablePredicates
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number, Variable


def test_comparing_two_numbers_is_answered_by_computing_it() -> None:
    assert EvaluablePredicates().holds(Literal("less", (Number(3), Number(4)))) is True


def test_a_comparison_that_does_not_hold_says_so() -> None:
    assert EvaluablePredicates().holds(Literal("less", (Number(4), Number(3)))) is False


def test_denying_a_comparison_answers_the_other_way() -> None:
    assert EvaluablePredicates().holds(Literal("less", (Number(4), Number(3)), True)) is True


def test_a_literal_still_holding_a_variable_is_unanswered_rather_than_false() -> None:
    assert EvaluablePredicates().holds(Literal("less", (Variable("N"), Number(4)))) is None


def test_a_predicate_nothing_computes_is_left_to_be_derived() -> None:
    predicates = EvaluablePredicates()

    assert predicates.holds(Literal("owns", (Constant("first"),))) is None
    assert not predicates.evaluable("owns")


def test_a_bound_further_up_makes_a_lower_one_add_nothing() -> None:
    predicates = EvaluablePredicates()
    eight = Literal("at_least", (Constant("long"), Number(8)))
    five = Literal("at_least", (Constant("long"), Number(5)))

    assert predicates.dominates(eight, five) and not predicates.dominates(five, eight)


def test_a_bound_further_down_makes_a_higher_one_add_nothing() -> None:
    predicates = EvaluablePredicates()
    five = Literal("at_most", (Constant("long"), Number(5)))
    eight = Literal("at_most", (Constant("long"), Number(8)))

    assert predicates.dominates(five, eight) and not predicates.dominates(eight, five)


def test_bounds_on_different_things_say_nothing_about_each_other() -> None:
    predicates = EvaluablePredicates()
    one = Literal("at_least", (Constant("long"), Number(8)))
    another = Literal("at_least", (Constant("short"), Number(5)))

    assert not predicates.dominates(one, another)


def test_a_game_s_own_predicate_is_registered_rather_than_built_in() -> None:
    plain = EvaluablePredicates()
    with_grid = plain.register("reaching", lambda thing, count: float(count) > 0)

    assert with_grid.evaluable("reaching") and not plain.evaluable("reaching")


def test_registering_leaves_the_predicates_it_was_given_untouched() -> None:
    plain = EvaluablePredicates()

    plain.register("reaching", lambda thing, count: True)

    assert not plain.evaluable("reaching")


def test_a_registered_predicate_is_answered_by_what_was_registered_for_it() -> None:
    predicates = EvaluablePredicates().register("even", lambda number: number % 2 == 0)

    assert predicates.holds(Literal("even", (Number(4),))) is True
    assert predicates.holds(Literal("even", (Number(3),))) is False


def test_a_registered_predicate_that_raises_leaves_the_literal_unanswered() -> None:
    def refuses(number: float) -> bool:
        raise ValueError("not for this")

    assert EvaluablePredicates().register("odd", refuses).holds(Literal("odd", (Number(3),))) is None
