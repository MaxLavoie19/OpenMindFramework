import pytest

from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.model.substitution import Substitution
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Number, Variable


def test_an_agreement_about_nothing_leaves_everything_as_it_was() -> None:
    literal = Literal("owns", (Variable("Player"), Constant("token")))

    assert Substitution().applied(literal) == literal


def test_a_variable_bound_is_replaced_wherever_it_is_read() -> None:
    agreed = Substitution().bound(Variable("Player"), Constant("first"))
    clause = Clause((Literal("wins", (Variable("Player"),)), Literal("owns", (Variable("Player"),), True)))

    applied = agreed.applied(clause)

    assert applied.ground


def test_binding_a_variable_to_another_that_is_already_bound_follows_it_through() -> None:
    agreed = Substitution().bound(Variable("A"), Variable("B")).bound(Variable("B"), Constant("first"))

    assert agreed.of(Variable("A")) == Constant("first")


def test_one_agreement_followed_by_another_says_what_both_together_say() -> None:
    first = Substitution().bound(Variable("A"), Variable("B"))
    second = Substitution().bound(Variable("B"), Constant("first"))

    both = first.then(second)

    assert both.of(Variable("A")) == Constant("first") and both.of(Variable("B")) == Constant("first")


def test_a_variable_nothing_was_agreed_about_stands_for_nothing() -> None:
    assert Substitution().of(Variable("Player")) is None


def test_a_variable_inside_a_function_is_replaced_too() -> None:
    agreed = Substitution().bound(Variable("Place"), Constant("corner"))
    term = Functor("next", (Variable("Place"),))

    assert agreed.applied(term) == Functor("next", (Constant("corner"),))


def test_a_clause_keeps_its_name_and_its_chance_through_an_agreement() -> None:
    agreed = Substitution().bound(Variable("Player"), Constant("first"))
    clause = Clause((Literal("wins", (Variable("Player"),)),), 0.8, "winning")

    applied = agreed.applied(clause)

    assert applied.probability == 0.8 and applied.name == "winning"


def test_a_budget_with_no_time_leaves_nothing_to_do() -> None:
    with pytest.raises(ValueError):
        InferenceBudget(0.0)


def test_a_budget_of_no_steps_leaves_nothing_to_do() -> None:
    with pytest.raises(ValueError):
        InferenceBudget(1.0, steps=0)


def test_a_budget_collects_more_than_one_proof_unless_told_otherwise() -> None:
    assert InferenceBudget(1.0).derivations is None


def test_a_number_is_left_alone_by_any_agreement() -> None:
    assert Substitution().bound(Variable("N"), Constant("first")).applied(Number(14)) == Number(14)
