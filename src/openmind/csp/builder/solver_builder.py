from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.csp.service.arc_consistency import ArcConsistency
from openmind.csp.service.backtracking_search import BacktrackingSearch
from openmind.csp.service.circuit_propagator import CircuitPropagator
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.csp.repository.solution_cache import SolutionCache
from openmind.csp.service.solver import Solver
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.service.rule_caller import RuleCaller
from openmind.rule.mapper.call_operand_mapper import CallOperandMapper


class SolverBuilder:
    """Wires a solver with its rule caller, call operand mapper, constraint checker, propagators, search and solution
    cache.

    **The caller may be given, because what a solver checks may be something OMF learned.** A constraint kept as
    a clause is answered by whatever the caller was built to answer clauses with, and a builder that makes its
    own caller cannot be told — so a game whose legality was induced would have every candidate raise at the
    moment it was checked. It is the same reason the effects runner takes one."""

    def __init__(self, rule_caller: RuleCaller | None = None) -> None:
        self._rule_caller = rule_caller

    def build(self) -> Solver:
        rule_caller = create_rule_caller() if self._rule_caller is None else self._rule_caller
        constraint_checker = ConstraintChecker(rule_caller)
        search = BacktrackingSearch(
            ArcConsistency(), AllDifferentPropagator(), CircuitPropagator(), constraint_checker
        )
        return Solver(rule_caller, CallOperandMapper(), constraint_checker, search, SolutionCache())
