from openmind.inference.service.resolver import Resolver
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Number, Variable


def fact(predicate: str, *arguments: object) -> Clause:
    return Clause((Literal(predicate, tuple(arguments)),))  # type: ignore[arg-type]


def rule(head: Literal, *body: Literal) -> Clause:
    return Clause((head, *(one.denied for one in body)))


def test_a_rule_against_a_fact_gives_a_new_fact() -> None:
    a_rule = rule(Literal("wins", (Variable("Player"),)), Literal("owns all", (Variable("Player"),)))
    a_fact = fact("owns all", Constant("first"))

    found = Resolver().resolve(a_rule, a_fact)

    assert len(found) == 1 and found[0][0].readable == "wins(first)"


def test_a_rule_against_a_rule_gives_a_new_rule() -> None:
    wins = rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),)))
    owns_all = rule(Literal("owns all", (Variable("Q"),)), Literal("took every place", (Variable("Q"),)))

    found = Resolver().resolve(wins, owns_all)

    assert len(found) == 1
    made = found[0][0]
    assert made.head is not None and made.head.predicate == "wins"
    assert made.body[0].predicate == "took every place"


def test_two_clauses_with_nothing_in_common_give_nothing() -> None:
    assert Resolver().resolve(fact("owns", Constant("first")), fact("holds", Constant("second"))) == ()


def test_a_fact_and_its_denial_give_the_contradiction() -> None:
    a_fact = fact("wins", Constant("first"))
    denied = Clause((Literal("wins", (Constant("first"),), True),))

    found = Resolver().resolve(a_fact, denied)

    assert len(found) == 1 and found[0][0].empty


def test_the_two_clauses_are_renamed_apart_so_the_same_letter_is_not_taken_for_one_thing() -> None:
    one = rule(Literal("beats", (Variable("X"), Variable("Y"))), Literal("stronger", (Variable("X"), Variable("Y"))))
    other = fact("stronger", Constant("first"), Constant("second"))

    found = Resolver().resolve(one, other)

    assert found[0][0].readable == "beats(first, second)"


def test_what_had_to_be_taken_to_mean_what_comes_back_with_the_conclusion() -> None:
    a_rule = rule(Literal("wins", (Variable("Player"),)), Literal("owns all", (Variable("Player"),)))

    _, agreed = Resolver().resolve(a_rule, fact("owns all", Constant("first")))[0]

    assert any(term == Constant("first") for _, term in agreed.bindings)


def test_a_conclusion_that_would_be_true_whatever_happens_is_not_drawn() -> None:
    one = Clause((Literal("a", ()), Literal("b", ())))
    other = Clause((Literal("a", (), True), Literal("b", (), True)))

    found = Resolver().resolve(one, other)

    assert found == ()


def test_a_clause_saying_the_same_thing_twice_of_agreeing_terms_says_it_once() -> None:
    clause = Clause((Literal("owns", (Variable("X"),)), Literal("owns", (Constant("first"),))))

    found = Resolver().factors(clause)

    assert len(found) == 1 and len(found[0][0].literals) == 1


def test_a_clause_with_nothing_to_collapse_is_left_alone() -> None:
    clause = Clause((Literal("owns", (Constant("first"),)), Literal("owns", (Constant("second"),))))

    assert Resolver().factors(clause) == ()


def test_a_chance_carries_through_a_step_as_both_clauses_together() -> None:
    a_rule = Clause((Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),), True)), 0.5)
    a_fact = Clause((Literal("owns all", (Constant("first"),)),), 0.5)

    found = Resolver().resolve(a_rule, a_fact)

    assert found[0][0].probability == 0.25


def test_a_condition_the_engine_can_work_out_is_discharged_when_it_is_met() -> None:
    clause = Clause((Literal("big", (Variable("N"),)), Literal("more", (Number(3), Number(1)), True)))

    settled = Resolver().settled(clause)

    assert settled is not None and settled.literals == (Literal("big", (Variable("N"),)),)


def test_a_rule_whose_condition_fails_is_idle_and_says_nothing_here() -> None:
    clause = Clause((Literal("big", (Variable("N"),)), Literal("more", (Number(0), Number(1)), True)))

    assert Resolver().settled(clause) is None


def test_a_condition_not_settled_enough_to_work_out_is_left_to_be_settled_later() -> None:
    clause = Clause((Literal("big", (Variable("N"),)), Literal("more", (Variable("N"), Number(1)), True)))

    settled = Resolver().settled(clause)

    assert settled is not None and len(settled.literals) == 2


def test_a_clause_of_things_the_engine_cannot_work_out_is_left_as_it_is() -> None:
    clause = Clause((Literal("owns", (Constant("first"),)),))

    assert Resolver().settled(clause) == clause
