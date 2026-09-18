import dataclasses
import logging
import time

from openmind.inference.constant.logic_constant import DISPROVED, INDEPENDENT, INDUCTION_RULE, INT, PROVED, UNKNOWN
from openmind.inference.mapper.formula_text_mapper import FormulaTextMapper
from openmind.inference.model.formula import And, Compare, Exists, ForAll, Formula, Implies
from openmind.inference.model.goal import Goal
from openmind.inference.model.proof import Proof
from openmind.inference.model.proof_step import ProofStep
from openmind.inference.model.prover import Prover
from openmind.inference.model.symbol import FunctionSymbol
from openmind.inference.model.term import Application, Number, Variable

logger = logging.getLogger(__name__)


class InductionProver:
    """Proves a statement over the natural numbers by induction, with a prover for each part. The goal's conclusion
    reads `∀n (n ≥ 0 → P(n))`, possibly with other variables beside `n`: the base `P(0)` and the step
    `∀n (n ≥ 0 ∧ P(n) → P(n + 1))` are proven in turn, the base with half the seconds. Both proven prove the conclusion,
    its steps the base's then the step's, then one `induction` step using the last of each. A base not proven decides:
    disproved or independent, the conclusion is too, since it holds at 0; otherwise unknown. A step not proven leaves the
    conclusion unknown, as a stronger statement might still be proven."""

    def __init__(self, prover: Prover, formula_text_mapper: FormulaTextMapper) -> None:
        self._prover = prover
        self._text = formula_text_mapper

    def prove(self, goal: Goal, variable: Variable, seconds: float) -> Proof:
        started = time.perf_counter()
        others, body = self._parts(goal.conclusion, variable)
        zero = self._replaced(body, variable, Number(0))
        successor = Application(FunctionSymbol("+", (INT, INT), INT), (variable, Number(1)))
        base_goal = dataclasses.replace(goal, conclusion=ForAll(others, zero) if others else zero)
        step_goal = dataclasses.replace(
            goal,
            conclusion=ForAll(
                (*others, variable),
                Implies(And((Compare(">=", variable, Number(0)), body)), self._replaced(body, variable, successor)),
            ),
        )
        conclusion = self._text.to_text(goal.conclusion)
        base = self._prover.prove(base_goal, seconds / 2)
        if base.status != PROVED:
            status = base.status if base.status in (DISPROVED, INDEPENDENT) else UNKNOWN
            spent = time.perf_counter() - started
            logger.info("Induction on %s couldn't prove %s: the base is %s", variable.name, conclusion, base.status)
            return Proof(goal, status, base.steps, spent, base.counterexample, f"the base is {base.status}")
        step = self._prover.prove(step_goal, seconds - (time.perf_counter() - started))
        spent = time.perf_counter() - started
        if step.status != PROVED:
            logger.info("Induction on %s couldn't prove %s: the step is %s", variable.name, conclusion, step.status)
            return Proof(goal, UNKNOWN, (), spent, None, f"the step is {step.status}")
        offset = len(base.steps)
        shifted = tuple(
            ProofStep(item.number + offset, item.rule, tuple(number + offset for number in item.premises), item.formula)
            for item in step.steps
        )
        last = ProofStep(offset + len(shifted) + 1, INDUCTION_RULE, (offset, offset + len(shifted)), goal.conclusion)
        logger.info(
            "Proved %s by induction on %s: base in %d steps, step in %d steps, %.3f seconds",
            conclusion,
            variable.name,
            len(base.steps),
            len(step.steps),
            spent,
        )
        return Proof(goal, PROVED, (*base.steps, *shifted, last), spent)

    def _parts(self, conclusion: Formula, variable: Variable) -> tuple[tuple[Variable, ...], Formula]:
        """The other variables of the conclusion and P, from `∀... (n ≥ 0 → P)`; any other shape raises ValueError."""
        if isinstance(conclusion, ForAll) and variable in conclusion.variables and variable.sort == INT:
            body = conclusion.body
            if isinstance(body, Implies) and body.premise == Compare(">=", variable, Number(0)):
                return tuple(item for item in conclusion.variables if item != variable), body.conclusion
        raise ValueError(f"Induction on {variable.name} needs a conclusion ∀{variable.name} ({variable.name} ≥ 0 → ...) over whole numbers")

    def _replaced(self, node: object, variable: Variable, value: object) -> object:
        """The formula or term with every free occurrence of the variable replaced by the value."""
        if node == variable:
            return value
        if isinstance(node, (ForAll, Exists)) and variable in node.variables:
            return node
        if isinstance(node, tuple):
            return tuple(self._replaced(item, variable, value) for item in node)
        if dataclasses.is_dataclass(node) and not isinstance(node, type) and not isinstance(node, Variable):
            changes = {field.name: self._replaced(getattr(node, field.name), variable, value) for field in dataclasses.fields(node)}
            return dataclasses.replace(node, **changes)
        return node
