from openmind.inference.model.example import Example
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.clause_challenger import ClauseChallenger
from openmind.inference.service.clause_learner import ClauseLearner
from openmind.rule.model.clause import Clause
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant

LEGAL = Literal("legal", ())


def case(holds: bool, where: object = None, **readings: object) -> Example:
    literals = tuple(Literal(name, (Constant(value),)) for name, value in readings.items())
    return Example(literals, holds, where)


def rule(**conditions: object) -> Clause:
    return Clause((LEGAL, *(Literal(name, (Constant(value),)).denied for name, value in conditions.items())))


def test_every_literal_a_clause_leans_on_is_something_to_look_for_an_exception_to() -> None:
    clause = rule(kind="walker", step=1)

    found = ClauseChallenger().challenges((clause,))

    assert len(found) == 2 and {one.predicate for _, one in found} == {"kind", "step"}


def test_a_challenge_asks_for_a_case_meeting_every_other_literal_and_failing_this_one() -> None:
    clause = rule(kind="walker", step=1)
    challenger = ClauseChallenger()
    _, literal = next(one for one in challenger.challenges((clause,)) if one[1].predicate == "step")

    widened = challenger.asked(clause, literal)

    assert {one.predicate for one in widened.body} == {"kind"}


def test_a_literal_turning_away_cases_that_hold_is_counted_against_itself() -> None:
    clause = rule(kind="walker", step=1)
    examples = [case(True, kind="walker", step=2)]

    counted = ClauseChallenger().challenged((clause,), examples)

    assert counted.get((Literal("step", (Constant(1),)), False)) == 1


def test_a_literal_failing_to_turn_away_what_does_not_hold_is_counted_for_itself() -> None:
    clause = rule(kind="walker")
    examples = [case(False, kind="walker")]

    counted = ClauseChallenger().challenged((clause,), examples)

    assert counted.get((Literal("kind", (Constant("walker"),)), True)) == 1


def test_a_clause_a_case_breaks_does_not_survive_it() -> None:
    clause = rule(kind="walker")

    surviving, broken = ClauseChallenger().refuted((clause,), [case(False, kind="walker")])

    assert surviving == () and len(broken) == 1


def test_a_clause_nothing_breaks_survives() -> None:
    clause = rule(kind="walker")

    surviving, broken = ClauseChallenger().refuted((clause,), [case(True, kind="walker")])

    assert surviving == (clause,) and broken == ()


def test_how_long_a_broken_clause_had_stood_comes_back_with_it() -> None:
    clause = rule(kind="walker")

    _, broken = ClauseChallenger().refuted((clause,), [case(False, kind="walker")], {clause.readable: 3_000})

    assert broken[0].stood == 3_000 and broken[0].stood_longer_than(100)


def test_a_clause_broken_early_is_a_guess_and_not_a_rare_rule_met() -> None:
    clause = rule(kind="walker")

    _, broken = ClauseChallenger().refuted((clause,), [case(False, kind="walker")], {clause.readable: 4})

    assert not broken[0].stood_longer_than(100)


def test_the_case_that_broke_a_clause_is_kept_so_later_clauses_can_be_made_to_face_it() -> None:
    clause = rule(kind="walker")
    breaking = case(False, "a position nobody will find again", kind="walker")

    _, broken = ClauseChallenger().refuted((clause,), [breaking])

    assert broken[0].broken_by is breaking and broken[0].broken_by.where == "a position nobody will find again"


def test_cases_that_do_not_hold_and_no_clause_rules_out_are_reported() -> None:
    examples = [case(False, kind="walker"), case(False, kind="flyer")]

    found = ClauseChallenger().unexplained((), examples)

    assert len(found) == 2


def test_nothing_is_reported_unexplained_where_a_clause_accounts_for_it() -> None:
    clause = rule(kind="walker")

    found = ClauseChallenger().unexplained((clause,), [case(False, kind="walker")])

    assert found == ()


def test_what_turned_out_possible_and_was_never_considered_is_reported() -> None:
    considered = (Literal("move", (Constant("a"),)), Literal("move", (Constant("b"),)))
    held = (Literal("move", (Constant("b"),)), Literal("move", (Constant("c"),)))

    found = ClauseChallenger().unforeseen(considered, held)

    assert found == (Literal("move", (Constant("c"),)),)


def test_an_outcome_nothing_predicted_is_a_clause_missing_rather_than_a_rate_to_correct() -> None:
    seen = [case(True, landed="somewhere new"), case(True, landed="somewhere new")]

    found = ClauseChallenger().mispredicted((), seen, InferenceBudget(5.0))

    assert found and all(one.unforeseen for one in found)


def test_an_outcome_predicted_at_the_wrong_rate_is_not_reported_as_a_missing_clause() -> None:
    clause = Clause((Literal("landed", ()), Literal("tried", (Constant("yes"),)).denied), 0.5)
    seen = [case(True, tried="yes", landed="here") for _ in range(10)]

    found = ClauseChallenger().mispredicted((clause,), seen, InferenceBudget(5.0))

    assert all(not one.unforeseen for one in found if one.outcome.predicate == "tried")
