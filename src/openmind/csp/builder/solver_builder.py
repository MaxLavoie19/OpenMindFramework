from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.csp.service.arc_consistency import ArcConsistency
from openmind.csp.service.backtracking_search import BacktrackingSearch
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.csp.service.solver import Solver
from openmind.rule.mapper.call_operand_mapper import CallOperandMapper
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class SolverBuilder:
    """Wires a solver with its rule compiler and runner, call operand mapper, constraint checker, propagators and
    search."""

    def build(self) -> Solver:
        rule_runner = RuleRunner(StateNamespaceMapper(VariableNameMapper()))
        constraint_checker = ConstraintChecker(rule_runner)
        search = BacktrackingSearch(ArcConsistency(), AllDifferentPropagator(), constraint_checker)
        return Solver(RuleCompiler(), rule_runner, CallOperandMapper(), constraint_checker, search)
