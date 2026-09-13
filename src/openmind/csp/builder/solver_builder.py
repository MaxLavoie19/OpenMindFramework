from openmind.csp.service.all_different_propagator import AllDifferentPropagator
from openmind.csp.service.arc_consistency import ArcConsistency
from openmind.csp.service.backtracking_search import BacktrackingSearch
from openmind.csp.service.constraint_checker import ConstraintChecker
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.mapper.parameter_scope_mapper import ParameterScopeMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class SolverBuilder:
    """Wires a solver with its constraint checker, scope mapper, propagators, search and text mapper."""

    def build(self) -> Solver:
        names = VariableNameMapper()
        constraint_checker = ConstraintChecker(Interpreter(names))
        search = BacktrackingSearch(ArcConsistency(), AllDifferentPropagator(), constraint_checker)
        return Solver(constraint_checker, ParameterScopeMapper(), search, ExpressionTextMapper(names))
