from openmind.epistemology.service.foundherentism import Foundherentism
from openmind.inference.constant.certainty_constant import CANNOT_BOTH_HOLD
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.coherence_checker import CoherenceChecker
from openmind.inference.service.justifier import Justifier
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number, Variable

WIDE = Constant("wide")


def checker() -> CoherenceChecker:
    return CoherenceChecker(Justifier(Foundherentism()))


def fact(predicate: str, *arguments: object, name: str = "") -> Clause:
    return Clause((Literal(predicate, tuple(arguments)),), 1.0, name)  # type: ignore[arg-type]


def test_clauses_that_can_all_hold_at_once_are_no_conflict() -> None:
    clauses = (fact("reaches", WIDE, Number(14)), fact("owned", WIDE))

    assert checker().contradictory(clauses, "a game", InferenceBudget(5.0)) == ()


def test_a_thing_said_and_denied_cannot_both_hold() -> None:
    clauses = (
        fact("owned", WIDE, name="what was seen"),
        Clause((Literal("owned", (WIDE,), True),), 1.0, "what the rule says"),
    )

    found = checker().contradictory(clauses, "a game", InferenceBudget(5.0))

    assert len(found) >= 1 and found[0].kind == CANNOT_BOTH_HOLD


def test_a_conflict_names_the_clauses_a_person_has_to_choose_between() -> None:
    clauses = (
        fact("owned", WIDE, name="what was seen"),
        Clause((Literal("owned", (WIDE,), True),), 1.0, "what the rule says"),
    )

    found = checker().contradictory(clauses, "a game", InferenceBudget(5.0))

    assert set(found[0].ids) == {"what was seen", "what the rule says"}


def test_a_contradiction_reached_through_a_rule_is_found_where_comparing_values_would_see_nothing() -> None:
    clauses = (
        fact("reaches", WIDE, Number(14), name="how far it reaches"),
        Clause(
            (
                Literal("modest", (Variable("N"),)),
                Literal("reaches", (Variable("T"), Variable("N")), True),
            ),
            1.0,
            "nothing reaches immodestly far",
        ),
        Clause((Literal("modest", (Number(14),), True),), 1.0, "fourteen is not modest"),
    )

    found = checker().contradictory(clauses, "a game", InferenceBudget(5.0))

    assert len(found) >= 1 and found[0].kind == CANNOT_BOTH_HOLD
    assert "how far it reaches" in found[0].ids and "fourteen is not modest" in found[0].ids


def test_a_conflict_is_reported_and_not_resolved() -> None:
    clauses = (
        fact("owned", WIDE, name="what was seen"),
        Clause((Literal("owned", (WIDE,), True),), 1.0, "what the rule says"),
    )

    found = checker().contradictory(clauses, "a game", InferenceBudget(5.0))

    assert all(conflict.id for conflict in found)
