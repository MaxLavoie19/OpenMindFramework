import pytest

from openmind.inference.constant.logic_constant import DISPROVED, INDEPENDENT, INT, PROVED, REAL, UNKNOWN
from openmind.inference.mapper.formula_text_mapper import FormulaTextMapper
from openmind.inference.mapper.z3_formula_mapper import Z3FormulaMapper
from openmind.inference.model.formula import And, Atom, Compare, Equal, Formula, Not, Truth
from openmind.inference.model.goal import Goal
from openmind.inference.model.sort import Sort
from openmind.inference.model.symbol import FunctionSymbol, PredicateSymbol
from openmind.inference.model.term import Application, Constant, Number
from openmind.inference.model.theory import Axiom
from openmind.inference.service.theory_library import TheoryLibrary
from openmind.inference.service.z3_prover import Z3Prover

pytestmark = pytest.mark.log_level("INFO")

LIBRARY = TheoryLibrary()
THING = Sort("Thing")
SETS = LIBRARY.set_sort(THING)
A, B, C = (Constant(name, SETS) for name in "ABC")
SUBSET = LIBRARY.subset(THING)
MEMBER = LIBRARY.member(THING)


def new_prover() -> Z3Prover:
    return Z3Prover(Z3FormulaMapper(), FormulaTextMapper())


def subset(first: Constant, second: Constant) -> Formula:
    return Atom(SUBSET, (first, second))


def test_a_conclusion_that_follows_is_proved_with_steps_ending_in_a_contradiction(caplog: pytest.LogCaptureFixture) -> None:
    goal = Goal(subset(A, C), (Axiom("A ⊆ B", subset(A, B)), Axiom("B ⊆ C", subset(B, C))), (LIBRARY.sets(THING),))

    proof = new_prover().prove(goal, 10.0)

    assert proof.status == PROVED and proof.counterexample is None
    assert proof.steps[-1].formula == Truth(False)
    assert [step.number for step in proof.steps] == list(range(1, len(proof.steps) + 1))
    assert all(premise < step.number for step in proof.steps for premise in step.premises)
    asserted = {step.formula for step in proof.steps if step.rule == "asserted"}
    assert {subset(A, B), subset(B, C), Not(subset(A, C))} <= asserted
    assert any(message.startswith(f"Proved A ⊆ C in {len(proof.steps)} steps, ") for message in caplog.messages)


def test_a_conclusion_whose_negation_follows_is_disproved() -> None:
    proof = new_prover().prove(Goal(Not(subset(A, A)), (), (LIBRARY.sets(THING),)), 10.0)

    assert proof.status == DISPROVED and proof.steps[-1].formula == Truth(False)


def test_a_conclusion_that_neither_follows_nor_is_refuted_is_independent_with_a_counterexample() -> None:
    goal = Goal(subset(B, A), (Axiom("A ⊆ B", subset(A, B)),), (LIBRARY.sets(THING),))

    proof = new_prover().prove(goal, 10.0)

    assert (proof.status, proof.steps) == (INDEPENDENT, ())
    assert proof.counterexample is not None and "A" in proof.counterexample


def test_a_goal_the_time_isn_t_enough_for_is_unknown_with_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    a, b, c = (Constant(name, INT) for name in "abc")
    cube = FunctionSymbol("*", (INT, INT), INT)

    def cubed(term: Constant) -> Application:
        return Application(cube, (Application(cube, (term, term)), term))

    plus = FunctionSymbol("+", (INT, INT), INT)
    fermat = And((Compare(">", a, Number(0)), Compare(">", b, Number(0)), Compare(">", c, Number(0))))
    goal = Goal(Not(Equal(Application(plus, (cubed(a), cubed(b))), cubed(c))), (Axiom("positive", fermat),))

    proof = new_prover().prove(goal, 0.001)

    assert proof.status == UNKNOWN and proof.reason
    assert any(message.startswith("Unknown ") for message in caplog.messages)


def test_a_selected_member_is_a_member_the_clause_holds_for() -> None:
    thing = LIBRARY.sets(THING).axioms[0].formula.body.right.variables[0]  # type: ignore[attr-defined]
    heavy = PredicateSymbol("heavy", (THING,))
    selection = LIBRARY.selection("heavy", thing, Atom(heavy, (thing,)))
    item = Constant("item", THING)
    selected = Application(selection.symbols[0], (A,))  # type: ignore[arg-type]
    goal = Goal(
        And((Atom(MEMBER, (item, A)), Atom(heavy, (item,)))),
        (Axiom("selected", Atom(MEMBER, (item, selected))),),
        (LIBRARY.sets(THING), selection),
    )

    assert new_prover().prove(goal, 10.0).status == PROVED


def test_the_best_and_the_worst_of_a_value_bound_every_member_s_value() -> None:
    weight = FunctionSymbol("weight", (THING,), REAL)
    best, worst = LIBRARY.best(weight), LIBRARY.worst(weight)
    item = Constant("item", THING)
    value = Application(weight, (item,))
    bounded = And(
        (
            Compare("<=", value, Application(best.symbols[0], (A,))),  # type: ignore[arg-type]
            Compare(">=", value, Application(worst.symbols[0], (A,))),  # type: ignore[arg-type]
        )
    )

    goal = Goal(bounded, (Axiom("member", Atom(MEMBER, (item, A))),), (LIBRARY.sets(THING), best, worst))

    assert new_prover().prove(goal, 10.0).status == PROVED
    with pytest.raises(ValueError, match="reads one member"):
        LIBRARY.best(FunctionSymbol("distance", (THING, THING), REAL))
