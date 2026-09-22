from openmind.inference.model.substitution import Substitution
from openmind.inference.service.unifier import Unifier
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Functor, Number, Variable


def test_two_literals_saying_the_same_of_the_same_things_agree_about_nothing() -> None:
    one = Literal("owns", (Constant("first"), Constant("token")))

    agreed = Unifier().unify(one, one)

    assert agreed == Substitution()


def test_a_variable_comes_to_stand_for_what_the_other_literal_says_there() -> None:
    general = Literal("owns", (Variable("Player"), Constant("token")))
    particular = Literal("owns", (Constant("first"), Constant("token")))

    agreed = Unifier().unify(general, particular)

    assert agreed is not None and agreed.of(Variable("Player")) == Constant("first")


def test_a_variable_used_twice_has_to_stand_for_the_same_thing_both_times() -> None:
    general = Literal("beats", (Variable("Side"), Variable("Side")))
    particular = Literal("beats", (Constant("first"), Constant("second")))

    agreed = Unifier().unify(general, particular)

    assert agreed is None


def test_two_literals_about_different_things_never_agree() -> None:
    one = Literal("owns", (Variable("Player"),))
    other = Literal("holds", (Variable("Player"),))

    agreed = Unifier().unify(one, other)

    assert agreed is None


def test_a_variable_is_never_made_to_stand_for_something_containing_itself() -> None:
    one = Literal("same", (Variable("Place"),))
    other = Literal("same", (Functor("next", (Variable("Place"),)),))

    agreed = Unifier().unify(one, other)

    assert agreed is None


def test_functions_agree_where_they_are_the_same_function_of_agreeing_terms() -> None:
    one = Literal("at", (Functor("next", (Variable("Place"),)),))
    other = Literal("at", (Functor("next", (Constant("corner"),)),))

    agreed = Unifier().unify(one, other)

    assert agreed is not None and agreed.of(Variable("Place")) == Constant("corner")


def test_matching_binds_only_the_general_literal_s_variables() -> None:
    general = Literal("owns", (Variable("Player"), Variable("Thing")))
    particular = Literal("owns", (Constant("first"), Variable("Thing")))

    agreed = Unifier().matches(general, particular)

    assert agreed is not None and agreed.of(Variable("Player")) == Constant("first")


def test_matching_refuses_to_change_the_particular_literal_to_suit_the_general_one() -> None:
    general = Literal("owns", (Constant("first"),))
    particular = Literal("owns", (Variable("Player"),))

    agreed = Unifier().matches(general, particular)

    assert agreed is None


def test_a_clause_renamed_shares_no_variable_with_the_one_it_came_from() -> None:
    clause = Clause((Literal("wins", (Variable("Player"),)), Literal("owns", (Variable("Player"),), True)))

    renamed = Unifier().renamed(clause, 1)

    assert set(renamed.variables).isdisjoint(clause.variables)


def test_a_clause_renamed_still_says_the_same_thing_of_the_same_variable_throughout() -> None:
    clause = Clause((Literal("wins", (Variable("Player"),)), Literal("owns", (Variable("Player"),), True)))

    renamed = Unifier().renamed(clause, 1)

    assert len(renamed.variables) == 1


def test_numbers_agree_only_with_the_same_number() -> None:
    unifier = Unifier()
    one = Literal("reaches", (Variable("Thing"), Number(14)))

    agreed = unifier.unify(one, Literal("reaches", (Constant("long"), Number(14))))
    refused = unifier.unify(one, Literal("reaches", (Constant("long"), Number(8))))

    assert agreed is not None and refused is None


def test_a_thing_denied_is_not_a_case_of_that_thing_asserted() -> None:
    asserted = Literal("owned", (Variable("Thing"),))
    denied = Literal("owned", (Constant("first"),), True)

    assert Unifier().matches(asserted, denied) is None


def test_unifying_still_ignores_denial_since_it_looks_for_literals_that_disagree() -> None:
    asserted = Literal("owned", (Variable("Thing"),))
    denied = Literal("owned", (Constant("first"),), True)

    assert Unifier().unify(asserted, denied) is not None
