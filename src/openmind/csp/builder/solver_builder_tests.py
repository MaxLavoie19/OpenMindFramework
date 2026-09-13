from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.csp.service.solver import Solver


def test_build_gives_a_solver() -> None:
    assert isinstance(SolverBuilder().build(), Solver)
