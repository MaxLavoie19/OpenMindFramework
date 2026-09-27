from openmind.inference.model.answer import DISPROVED, PROVED, UNKNOWN
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.goal_prover import NOTHING_BEARS, GoalProver
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number, Variable

FIRST, WIDE, NARROW = Constant("first"), Constant("wide"), Constant("narrow")


def fact(predicate: str, *arguments: object, name: str = "", probability: float = 1.0) -> Clause:
    return Clause((Literal(predicate, tuple(arguments)),), probability, name)  # type: ignore[arg-type]


def rule(head: Literal, *body: Literal, name: str = "") -> Clause:
    return Clause((head, *(one.denied for one in body)), 1.0, name)


def test_a_question_the_clauses_settle_is_proved() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", FIRST),
    )

    answer = GoalProver().ask(clauses, fact("wins", FIRST), InferenceBudget(5.0))

    assert answer.status == PROVED


def test_a_question_nothing_bears_on_is_unknown_rather_than_refused() -> None:
    answer = GoalProver().ask((fact("owns all", FIRST),), fact("wins", FIRST), InferenceBudget(5.0))

    assert answer.status == UNKNOWN and answer.reason == NOTHING_BEARS


def test_a_question_the_clauses_contradict_is_disproved_which_is_more_than_unproved() -> None:
    clauses = (
        rule(Literal("safe", (Variable("P"),), True), Literal("at risk", (Variable("P"),))),
        fact("at risk", FIRST),
    )

    answer = GoalProver().ask(clauses, fact("safe", FIRST), InferenceBudget(5.0))

    assert answer.status == DISPROVED


def test_every_proof_is_collected_rather_than_the_first() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),)), name="one way"),
        rule(Literal("wins", (Variable("P"),)), Literal("others resigned", (Variable("P"),)), name="another way"),
        fact("owns all", FIRST),
        fact("others resigned", FIRST),
    )

    answer = GoalProver().ask(clauses, fact("wins", FIRST), InferenceBudget(5.0))

    assert len(answer.derivations) >= 2


def test_a_proof_says_which_clauses_it_rests_on() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),)), name="taking everything wins"),
        fact("owns all", FIRST, name="what was seen"),
    )

    answer = GoalProver().ask(clauses, fact("wins", FIRST), InferenceBudget(5.0))

    assert set(answer.derivations[0].rests_on) >= {"taking everything wins", "what was seen"}


def test_a_question_answered_through_a_chain_of_rules_is_still_proved() -> None:
    clauses = (
        rule(Literal("worth at least", (Variable("T"), Variable("N"))), Literal("affords", (Variable("T"), Variable("N")))),
        rule(
            Literal("worth at least", (Variable("T"), Variable("N"))),
            Literal("takes in", (Variable("T"), Variable("O"))),
            Literal("worth at least", (Variable("O"), Variable("N"))),
        ),
        fact("affords", NARROW, Number(14)),
        fact("takes in", WIDE, NARROW),
    )

    answer = GoalProver().ask(clauses, fact("worth at least", WIDE, Number(14)), InferenceBudget(5.0))

    assert answer.status == PROVED


def test_a_question_asked_about_any_thing_at_all_is_answered_the_same_way() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", FIRST),
    )

    answer = GoalProver().ask(clauses, fact("wins", Variable("Who")), InferenceBudget(5.0))

    assert answer.status == PROVED


def test_a_question_a_computed_condition_rules_out_is_not_proved() -> None:
    clauses = (
        rule(
            Literal("cannot all escape", (Variable("T"),)),
            Literal("at stake", (Variable("T"), Variable("N"))),
            Literal("more", (Variable("N"), Number(1))),
        ),
        fact("at stake", NARROW, Number(1)),
    )

    answer = GoalProver().ask(clauses, fact("cannot all escape", NARROW), InferenceBudget(5.0))

    assert answer.status != PROVED


def test_a_question_a_computed_condition_allows_is_proved() -> None:
    clauses = (
        rule(
            Literal("cannot all escape", (Variable("T"),)),
            Literal("at stake", (Variable("T"), Variable("N"))),
            Literal("more", (Variable("N"), Number(1))),
        ),
        fact("at stake", WIDE, Number(2)),
    )

    answer = GoalProver().ask(clauses, fact("cannot all escape", WIDE), InferenceBudget(5.0))

    assert answer.status == PROVED


def test_a_budget_of_a_single_derivation_stops_once_it_has_one() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        rule(Literal("wins", (Variable("P"),)), Literal("others resigned", (Variable("P"),))),
        fact("owns all", FIRST),
        fact("others resigned", FIRST),
    )

    answer = GoalProver().ask(clauses, fact("wins", FIRST), InferenceBudget(5.0, derivations=1))

    assert answer.status == PROVED and len(answer.derivations) == 1


def test_a_shallow_answer_is_not_left_behind_a_deep_one() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        rule(Literal("owns all", (Variable("P"),)), Literal("took every place", (Variable("P"),)), name="the long way"),
        fact("owns all", FIRST, name="the short way"),
        fact("took every place", FIRST),
    )

    answer = GoalProver().ask(clauses, fact("wins", FIRST), InferenceBudget(5.0, derivations=1))

    rests_on = answer.derivations[0].rests_on

    assert "the short way" in rests_on and "the long way" not in rests_on


def test_an_answer_says_how_long_it_took() -> None:
    answer = GoalProver().ask((fact("owns all", FIRST),), fact("owns all", FIRST), InferenceBudget(5.0))

    assert answer.seconds >= 0.0 and answer.proved


def test_a_conclusion_leaning_on_a_doubtful_clause_carries_which_one() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", FIRST, name="what was seen", probability=0.8),
    )

    answer = GoalProver().ask(clauses, fact("wins", FIRST), InferenceBudget(5.0))

    assert answer.derivations[0].chances == ("what was seen",)


def test_a_proved_answer_carries_how_likely_it_is() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", FIRST),
    )

    answer = GoalProver().ask(clauses, fact("wins", FIRST), InferenceBudget(5.0))

    assert answer.chance is not None and answer.chance.value == 1.0


def test_an_answer_reached_only_through_a_doubtful_clause_is_no_surer_than_that_clause() -> None:
    clauses = (
        rule(Literal("wins", (Variable("P"),)), Literal("owns all", (Variable("P"),))),
        fact("owns all", FIRST, name="what was seen", probability=0.8),
    )

    answer = GoalProver().ask(clauses, fact("wins", FIRST), InferenceBudget(5.0))

    assert answer.chance is not None and abs(answer.chance.value - 0.8) < 1e-9
