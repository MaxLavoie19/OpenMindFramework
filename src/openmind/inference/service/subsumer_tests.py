from openmind.inference.service.subsumer import Subsumer
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Number, Variable


def a_rule(*literals: Literal) -> Clause:
    return Clause(literals)


def test_a_clause_asking_less_says_everything_a_clause_asking_more_says() -> None:
    asking_less = a_rule(Literal("legal", (Variable("Move"),)), Literal("straight", (Variable("Move"),), True))
    asking_more = a_rule(
        Literal("legal", (Variable("Move"),)),
        Literal("straight", (Variable("Move"),), True),
        Literal("near", (Variable("Move"),), True),
    )

    assert Subsumer().subsumes(asking_less, asking_more)


def test_a_clause_asking_more_does_not_say_what_a_clause_asking_less_says() -> None:
    asking_less = a_rule(Literal("legal", (Variable("Move"),)), Literal("straight", (Variable("Move"),), True))
    asking_more = a_rule(
        Literal("legal", (Variable("Move"),)),
        Literal("straight", (Variable("Move"),), True),
        Literal("near", (Variable("Move"),), True),
    )

    assert not Subsumer().subsumes(asking_more, asking_less)


def test_a_clause_with_a_variable_says_what_the_same_clause_about_one_thing_says() -> None:
    about_anything = a_rule(Literal("affords", (Variable("Thing"),)))
    about_one = a_rule(Literal("affords", (Constant("long"),)))

    assert Subsumer().subsumes(about_anything, about_one)


def test_a_clause_about_one_thing_does_not_say_what_a_clause_about_anything_says() -> None:
    about_anything = a_rule(Literal("affords", (Variable("Thing"),)))
    about_one = a_rule(Literal("affords", (Constant("long"),)))

    assert not Subsumer().subsumes(about_one, about_anything)


def test_a_variable_cannot_stand_for_two_things_to_make_a_clause_fit() -> None:
    one_thing = a_rule(Literal("pairs", (Variable("Thing"), Variable("Thing"))))
    two_things = a_rule(Literal("pairs", (Constant("first"), Constant("second"))))

    assert not Subsumer().subsumes(one_thing, two_things)


def test_a_stronger_bound_makes_a_weaker_one_add_nothing() -> None:
    at_least_eight = a_rule(Literal("at_least", (Constant("long"), Number(8))))
    at_least_five = a_rule(Literal("at_least", (Constant("long"), Number(5))))

    assert Subsumer().subsumes(at_least_eight, at_least_five)


def test_a_weaker_bound_does_not_make_a_stronger_one_add_nothing() -> None:
    at_least_eight = a_rule(Literal("at_least", (Constant("long"), Number(8))))
    at_least_five = a_rule(Literal("at_least", (Constant("long"), Number(5))))

    assert not Subsumer().subsumes(at_least_five, at_least_eight)


def test_a_bound_on_one_thing_says_nothing_about_a_bound_on_another() -> None:
    about_one = a_rule(Literal("at_least", (Constant("long"), Number(8))))
    about_another = a_rule(Literal("at_least", (Constant("short"), Number(5))))

    assert not Subsumer().subsumes(about_one, about_another)


def test_bounds_the_other_way_round_are_ordered_the_other_way_round() -> None:
    at_most_five = a_rule(Literal("at_most", (Constant("long"), Number(5))))
    at_most_eight = a_rule(Literal("at_most", (Constant("long"), Number(8))))

    assert Subsumer().subsumes(at_most_five, at_most_eight)


def test_the_contradiction_says_everything() -> None:
    assert Subsumer().subsumes(Clause(), a_rule(Literal("anything", ())))


def test_one_set_of_clauses_takes_in_another_where_each_of_the_other_s_is_said_by_one_of_these() -> None:
    wider = (
        a_rule(Literal("legal", (Variable("Move"),)), Literal("straight", (Variable("Move"),), True)),
        a_rule(Literal("legal", (Variable("Move"),)), Literal("diagonal", (Variable("Move"),), True)),
    )
    narrower = (a_rule(Literal("legal", (Variable("Move"),)), Literal("straight", (Variable("Move"),), True)),)

    subsumer = Subsumer()

    assert subsumer.within(wider, narrower) and not subsumer.within(narrower, wider)


def test_what_one_thing_affords_is_ordered_against_another_from_the_clauses_alone() -> None:
    straight = a_rule(Literal("moves", (Variable("Move"),)), Literal("straight", (Variable("Move"),), True))
    diagonal = a_rule(Literal("moves", (Variable("Move"),)), Literal("diagonal", (Variable("Move"),), True))
    anyhow = a_rule(Literal("moves", (Variable("Move"),)))

    ordered = Subsumer().ordered({"both": (anyhow,), "straight": (straight,), "diagonal": (diagonal,)})

    assert set(ordered) == {("both", "straight"), ("both", "diagonal")}


def test_two_clauses_asking_contradictory_things_cannot_hold_of_the_same_thing() -> None:
    asks = a_rule(Literal("empty", (Variable("Place"),), True))
    denies = a_rule(Literal("empty", (Variable("Place"),)))

    assert not Subsumer().compatible(asks, denies)


def test_two_clauses_speaking_of_different_things_are_left_compatible() -> None:
    one = a_rule(Literal("empty", (Variable("Place"),), True))
    other = a_rule(Literal("owned", (Variable("Place"),), True))

    assert Subsumer().compatible(one, other)


def test_a_clause_already_held_makes_a_weaker_one_redundant() -> None:
    held = (a_rule(Literal("at_least", (Constant("long"), Number(8)))),)

    assert Subsumer().redundant(a_rule(Literal("at_least", (Constant("long"), Number(5)))), held)
