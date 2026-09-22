from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant, Number, Variable


def test_a_clause_of_one_positive_literal_is_a_fact_it_states() -> None:
    clause = Clause((Literal("acts once", ()),))

    assert clause.definite and clause.head is not None and not clause.body


def test_a_clause_of_one_positive_and_some_denials_reads_as_a_rule() -> None:
    clause = Clause((Literal("wins", (Variable("Player"),)), Literal("owns all", (Variable("Player"),), True)))

    assert clause.definite and clause.body[0].predicate == "owns all" and not clause.body[0].negated


def test_a_clause_with_no_positive_literal_is_a_question_and_has_no_head() -> None:
    clause = Clause((Literal("wins", (Constant("first"),), True),))

    assert clause.head is None and not clause.definite


def test_a_clause_concluding_several_things_has_no_single_head() -> None:
    clause = Clause((Literal("wins", (Constant("first"),)), Literal("wins", (Constant("second"),))))

    assert clause.head is None and not clause.definite


def test_the_clause_with_no_literals_is_the_contradiction() -> None:
    assert Clause().empty


def test_a_clause_with_nothing_standing_for_anything_is_ground() -> None:
    assert Clause((Literal("reaches", (Constant("long"), Number(14))),)).ground


def test_a_clause_with_a_variable_in_it_is_not_ground() -> None:
    assert not Clause((Literal("reaches", (Variable("Thing"), Number(14))),)).ground


def test_a_clause_reads_back_the_variables_it_uses_each_once() -> None:
    clause = Clause((Literal("wins", (Variable("Player"),)), Literal("owns", (Variable("Player"),), True)))

    assert clause.variables == (Variable("Player"),)


def test_a_clause_that_always_holds_is_certain() -> None:
    assert Clause((Literal("acts once", ()),)).certain


def test_a_clause_that_holds_some_of_the_time_is_not_certain() -> None:
    assert not Clause((Literal("acts once", ()),), 0.8).certain


def test_a_rule_reads_as_its_head_then_what_it_asks_for() -> None:
    clause = Clause((Literal("wins", (Variable("Player"),)), Literal("owns all", (Variable("Player"),), True)))

    assert clause.readable == "wins(Player) :- owns all(Player)"


def test_a_clause_that_holds_some_of_the_time_says_so_when_read() -> None:
    clause = Clause((Literal("wins", (Variable("Player"),)),), 0.8)

    assert clause.readable == "0.8::wins(Player)"


def test_a_literal_knows_which_variables_it_reads() -> None:
    literal = Literal("between", (Variable("From"), Variable("To"), Variable("From")))

    assert literal.variables == (Variable("From"), Variable("To"))


def test_a_literal_and_its_denial_oppose_each_other() -> None:
    literal = Literal("owned", (Variable("Thing"),))

    assert literal.opposes(literal.denied) and not literal.opposes(literal)


def test_two_literals_saying_different_things_never_oppose_each_other() -> None:
    assert not Literal("owned", ()).opposes(Literal("held", (), True))
