from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.csp.service.solver import Solver


def create_solver() -> Solver:
    """A solver with propagation, backtracking search and its cache."""
    return SolverBuilder().build()
