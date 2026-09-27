from openmind.inference.mapper.clause_text_mapper import ClauseTextMapper
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number, Variable


def test_a_fact_reads_as_the_thing_it_says() -> None:
    clause = Clause((Literal("acts once", ()),))

    assert ClauseTextMapper().to_text(clause) == "acts once"


def test_a_fact_about_things_names_them_after_what_is_said_of_them() -> None:
    clause = Clause((Literal("reaches", (Constant("long"), Number(14))),))

    assert ClauseTextMapper().to_text(clause) == "reaches long and 14"


def test_a_rule_reads_as_its_conclusion_then_what_it_asks_for() -> None:
    clause = Clause((
        Literal("wins", (Variable("Player"),)),
        Literal("owns all", (Variable("Player"),), True),
    ))

    assert ClauseTextMapper().to_text(clause) == "wins any player where owns all any player"


def test_a_rule_asking_several_things_joins_them_with_and() -> None:
    clause = Clause((
        Literal("wins", (Variable("Player"),)),
        Literal("acts once", (), True),
        Literal("owns all", (Variable("Player"),), True),
    ))

    said = ClauseTextMapper().to_text(clause)

    assert said == "wins any player where acts once and owns all any player"


def test_a_rule_that_only_sometimes_holds_says_how_often() -> None:
    clause = Clause((Literal("wins", (Variable("Player"),)),), 0.8)

    assert ClauseTextMapper().to_text(clause) == "wins any player, 0.8 of the time"


def test_a_denial_reads_as_a_denial() -> None:
    assert ClauseTextMapper().literal_to_text(Literal("owned", (Constant("first"),), True)) == "not owned first"


def test_the_contradiction_says_that_nothing_can_hold() -> None:
    assert ClauseTextMapper().to_text(Clause()) == "nothing can hold"


def test_a_goal_reads_as_a_question() -> None:
    clause = Clause((Literal("wins", (Constant("first"),), True),))

    assert ClauseTextMapper().to_text(clause) == "is there wins first?"


def test_a_variable_reads_by_its_sort_where_the_game_gave_one() -> None:
    assert ClauseTextMapper().term_to_text(Variable("X", "player")) == "any player"


def test_a_variable_with_no_sort_reads_by_its_own_name() -> None:
    assert ClauseTextMapper().term_to_text(Variable("Thing")) == "any thing"


def test_a_variable_renamed_apart_still_reads_by_the_name_it_started_with() -> None:
    assert ClauseTextMapper().term_to_text(Variable("Thing#3")) == "any thing"


def test_nothing_standing_somewhere_reads_as_nothing() -> None:
    assert ClauseTextMapper().term_to_text(Constant(None)) == "nothing"
