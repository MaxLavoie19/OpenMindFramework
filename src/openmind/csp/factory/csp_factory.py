from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.csp.service.solver import Solver
from openmind.rule.service.rule_caller import RuleCaller


def create_solver(rule_caller: RuleCaller | None = None) -> Solver:
    """A solver with propagation, backtracking search and its cache, and with the caller it is given where what it
    checks is something OMF learned rather than something a project wrote."""
    return SolverBuilder(rule_caller).build()
