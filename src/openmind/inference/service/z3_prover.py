import logging
import time

import z3

from openmind.inference.constant.logic_constant import DISPROVED, INDEPENDENT, PROOF_BIND, PROVED, UNKNOWN
from openmind.inference.mapper.formula_text_mapper import FormulaTextMapper
from openmind.inference.mapper.z3_formula_mapper import Z3FormulaMapper
from openmind.inference.model.formula import Formula, Not
from openmind.inference.model.goal import Goal
from openmind.inference.model.proof import Proof
from openmind.inference.model.proof_step import ProofStep
from openmind.inference.model.sort import Sort
from openmind.inference.model.symbol import FunctionSymbol

logger = logging.getLogger(__name__)


class Z3Prover:
    """Proves goals with Z3, by refutation: the premises and the theories' axioms with the conclusion denied. A
    contradiction proves the conclusion, and the steps are Z3's proof of it. Otherwise the same is tried with the
    conclusion itself: a contradiction disproves it; a model both times makes it independent, the first model its
    counterexample; anything else is unknown. Each goal is proven in a context of its own, within the seconds given, the
    denied conclusion getting at most half of them so the conclusion itself still gets a try."""

    def __init__(self, z3_formula_mapper: Z3FormulaMapper, formula_text_mapper: FormulaTextMapper) -> None:
        self._mapper = z3_formula_mapper
        self._text = formula_text_mapper

    def prove(self, goal: Goal, seconds: float) -> Proof:
        started = time.perf_counter()
        context = z3.Context(proof=True)
        facts = [
            self._mapper.to_z3(axiom.formula, context)
            for axiom in (*goal.premises, *(axiom for theory in goal.theories for axiom in theory.axioms))
        ]
        sorts = self._sorts(goal)
        denied, denied_model, denied_reason = self._check(facts, Not(goal.conclusion), context, started, seconds / 2, sorts)
        conclusion = self._text.to_text(goal.conclusion)
        if denied is not None:
            proof = Proof(goal, PROVED, denied, time.perf_counter() - started)
            logger.info("Proved %s in %d steps, %.3f seconds", conclusion, len(denied), proof.seconds)
            return proof
        affirmed, affirmed_model, affirmed_reason = self._check(facts, goal.conclusion, context, started, seconds, sorts)
        spent = time.perf_counter() - started
        if affirmed is not None:
            logger.info("Disproved %s in %d steps, %.3f seconds", conclusion, len(affirmed), spent)
            return Proof(goal, DISPROVED, affirmed, spent)
        if denied_model is not None and affirmed_model is not None:
            logger.info("Independent %s, %.3f seconds: counterexample %s", conclusion, spent, denied_model)
            return Proof(goal, INDEPENDENT, (), spent, denied_model)
        reason = denied_reason or affirmed_reason
        logger.info("Unknown %s after %.3f seconds: %s", conclusion, spent, reason)
        return Proof(goal, UNKNOWN, (), spent, denied_model, reason)

    def _check(
        self, facts: list[z3.BoolRef], formula: Formula, context: z3.Context, started: float, seconds: float, sorts: tuple[Sort, ...]
    ) -> tuple[tuple[ProofStep, ...] | None, str | None, str | None]:
        """The proof steps when the facts and the formula contradict each other; otherwise the model when they have one, or
        why the prover couldn't tell."""
        left = seconds - (time.perf_counter() - started)
        if left <= 0.0:
            return None, None, "the time ran out"
        solver = z3.Solver(ctx=context)
        solver.set("timeout", max(1, int(left * 1000)))
        solver.add(*facts, self._mapper.to_z3(formula, context))
        result = solver.check()
        if result == z3.unsat:
            return self._steps(solver.proof(), sorts), None, None
        if result == z3.sat:
            return None, str(solver.model()), None
        return None, None, solver.reason_unknown()

    def _steps(self, proof: z3.ExprRef, sorts: tuple[Sort, ...]) -> tuple[ProofStep, ...]:
        """Z3's proof as numbered steps, every premise before the step using it, each proof shared once."""
        numbers: dict[int, int] = {}
        steps: list[ProofStep] = []
        stack: list[tuple[z3.ExprRef, bool]] = [(self._proof(proof), False)]
        while stack:
            node, expanded = stack.pop()
            if node.get_id() in numbers:
                continue
            children = node.children()
            premises = [self._proof(child) for child in children if self._is_proof(child)]
            if not expanded:
                stack.append((node, True))
                stack.extend((premise, False) for premise in reversed(premises) if premise.get_id() not in numbers)
                continue
            conclusion = children[-1] if children and not self._is_proof(children[-1]) else node
            number = len(steps) + 1
            numbers[node.get_id()] = number
            steps.append(
                ProofStep(
                    number,
                    node.decl().name(),
                    tuple(numbers[premise.get_id()] for premise in premises),
                    self._mapper.from_z3(conclusion, sorts),
                )
            )
        for step in steps:
            logger.debug("Step %d, %s from %s: %s", step.number, step.rule, list(step.premises), self._text.to_text(step.formula))
        return tuple(steps)

    def _is_proof(self, node: z3.ExprRef) -> bool:
        """Whether a node is a proof, or a proof under a binder, which Z3 writes as a lambda giving proofs."""
        sort = node.sort()
        return sort.name() == "Proof" or (sort.kind() == z3.Z3_ARRAY_SORT and sort.range().name() == "Proof")

    def _proof(self, node: z3.ExprRef) -> z3.ExprRef:
        """The proof itself, out of the lambdas and `proof-bind`s binding it, its bound variables left free."""
        while True:
            if z3.is_quantifier(node) and node.is_lambda():
                node = node.body()
            elif z3.is_app(node) and node.decl().name() == PROOF_BIND and node.num_args() == 1:
                node = node.arg(0)
            else:
                return node

    def _sorts(self, goal: Goal) -> tuple[Sort, ...]:
        """The sorts the goal's theories' symbols read and give, so a proof's formulas come back with them."""
        found: dict[str, Sort] = {}
        for theory in goal.theories:
            for symbol in theory.symbols:
                for sort in (*symbol.arguments, *((symbol.result,) if isinstance(symbol, FunctionSymbol) else ())):
                    found.setdefault(sort.name, sort)
        return tuple(found.values())
