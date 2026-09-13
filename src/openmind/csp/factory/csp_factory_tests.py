from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.service.solver import Solver


def test_create_solver_gives_a_solver() -> None:
    assert isinstance(create_solver(), Solver)
