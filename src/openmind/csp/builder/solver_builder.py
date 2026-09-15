from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.csp.service.arc_consistency import ArcConsistency
from openmind.csp.service.backtracking_search import BacktrackingSearch
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.csp.service.solver import Solver
from openmind.rule.factory.rule_factory import create_rule_caller
from openmind.rule.mapper.call_operand_mapper import CallOperandMapper


class SolverBuilder:
    """Wires a solver with its rule compiler and runner, call operand mapper, constraint checker, propagators and
    search."""

    def build(self) -> Solver:
        rule_caller = create_rule_caller()
        constraint_checker = ConstraintChecker(rule_caller)
        search = BacktrackingSearch(ArcConsistency(), AllDifferentPropagator(), constraint_checker)
        return Solver(rule_caller, CallOperandMapper(), constraint_checker, search)
