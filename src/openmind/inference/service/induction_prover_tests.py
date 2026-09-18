import pytest

from openmind.inference.constant.logic_constant import DISPROVED, INDUCTION_RULE, INT, PROVED, UNKNOWN
from openmind.inference.mapper.formula_text_mapper import FormulaTextMapper
from openmind.inference.mapper.z3_formula_mapper import Z3FormulaMapper
from openmind.inference.model.formula import Compare, Equal, ForAll, Implies
from openmind.inference.model.goal import Goal
from openmind.inference.model.symbol import FunctionSymbol
from openmind.inference.model.term import Application, Number, Variable
from openmind.inference.model.theory import Axiom
from openmind.inference.service.induction_prover import InductionProver
from openmind.inference.service.z3_prover import Z3Prover

pytestmark = pytest.mark.log_level("INFO")

N = Variable("n", INT)
F = FunctionSymbol("f", (INT,), INT)
PLUS, TIMES = FunctionSymbol("+", (INT, INT), INT), FunctionSymbol("*", (INT, INT), INT)
#: f(0) = 0 and f(n + 1) = f(n) + 2.
DEFINITION = (
    Axiom("f at 0", Equal(Application(F, (Number(0),)), Number(0))),
    Axiom("f after n", ForAll((N,), Equal(Application(F, (Application(PLUS, (N, Number(1))),)), Application(PLUS, (Application(F, (N,)), Number(2)))))),
)


def doubled(start: int) -> ForAll:
    """∀n (n ≥ 0 → f(n) = 2 × n + start)."""
    value = Application(PLUS, (Application(TIMES, (Number(2), N)), Number(start)))
    return ForAll((N,), Implies(Compare(">=", N, Number(0)), Equal(Application(F, (N,)), value)))


def new_prover() -> InductionProver:
    text = FormulaTextMapper()
    return InductionProver(Z3Prover(Z3FormulaMapper(), text), text)


def test_a_statement_over_the_naturals_is_proved_from_its_base_and_its_step(caplog: pytest.LogCaptureFixture) -> None:
    goal = Goal(doubled(0), DEFINITION)

    proof = new_prover().prove(goal, N, 20.0)

    assert proof.status == PROVED
    last = proof.steps[-1]
    assert (last.rule, last.formula, len(last.premises)) == (INDUCTION_RULE, doubled(0), 2)
    assert [step.number for step in proof.steps] == list(range(1, len(proof.steps) + 1))
    assert any(message.startswith("Proved ∀n ") and " by induction on n: base in " in message for message in caplog.messages)


def test_z3_alone_can_t_prove_what_induction_does() -> None:
    text = FormulaTextMapper()

    assert Z3Prover(Z3FormulaMapper(), text).prove(Goal(doubled(0), DEFINITION), 2.0).status == UNKNOWN


def test_a_base_that_doesn_t_hold_decides_and_a_conclusion_of_another_shape_is_rejected() -> None:
    proof = new_prover().prove(Goal(doubled(1), DEFINITION), N, 8.0)

    assert (proof.status, proof.reason) == (DISPROVED, "the base is disproved")
    with pytest.raises(ValueError, match="needs a conclusion"):
        new_prover().prove(Goal(Equal(Application(F, (N,)), Number(0)), DEFINITION), N, 1.0)
