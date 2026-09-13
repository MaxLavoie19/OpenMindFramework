import pytest

from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

pytestmark = pytest.mark.log_level("INFO")

SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def test_the_single_solution_fills_the_known_grid_and_pays_one() -> None:
    domain = create_sudoku_domain()
    names = VariableNameMapper()
    predictor = Predictor(Interpreter(names), names, ExpressionTextMapper(names), ActionTextMapper())

    actions = create_solver().solve(domain.problem, domain.initial_state, limit=2)

    assert len(actions) == 1
    ((outcome, probability),) = predictor.predict(domain.transitions, domain.initial_state, actions[0]).outcomes
    values = dict(outcome.variables)
    assert "".join(str(values[f"cell({row},{col})"]) for row in range(1, 10) for col in range(1, 10)) == SOLUTION
    assert (probability, values["payoff"]) == (1.0, 1.0)


def test_searching_without_a_limit_finds_no_other_solution() -> None:
    domain = create_sudoku_domain()

    assert len(create_solver().solve(domain.problem, domain.initial_state)) == 1
