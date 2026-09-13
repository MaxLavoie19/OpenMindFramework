import random

import pytest

from openmind.agent.constant.tictactoe_constant import VARIANTS
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

pytestmark = pytest.mark.log_level("INFO")


def every_line(width: int, height: int, length: int) -> list[list[tuple[int, int]]]:
    """Every run of length cells across, down, or along either diagonal, found without the factory's rules."""
    return [
        [(row + index * row_step, col + index * col_step) for index in range(length)]
        for row in range(1, height + 1)
        for col in range(1, width + 1)
        for row_step, col_step in ((0, 1), (1, 0), (1, 1), (1, -1))
        if 1 <= row + (length - 1) * row_step <= height and 1 <= col + (length - 1) * col_step <= width
    ]


@pytest.mark.parametrize(("name", "games"), [("standard", 50), ("fourinarow", 30), ("gomoku", 2)])
def test_random_games_follow_the_rules_after_every_move(name: str, games: int) -> None:
    variant = VARIANTS[name]
    domain = create_tictactoe_domain(variant)
    lines = every_line(variant.width, variant.height, variant.line)
    names = VariableNameMapper()
    solver = create_solver()
    predictor = Predictor(Interpreter(names), names, ExpressionTextMapper(names), ActionTextMapper())
    rng = random.Random(1)

    for _ in range(games):
        state = domain.initial_state
        while actions := solver.solve(domain.problem, state):
            ((state, _),) = predictor.predict(domain.transitions, state, rng.choice(actions)).outcomes
            values = dict(state.variables)
            grid = {
                (row, col): values[f"cell({row},{col})"]
                for row in range(1, variant.height + 1)
                for col in range(1, variant.width + 1)
            }
            marks = list(grid.values())
            winners = {grid[line[0]] for line in lines if grid[line[0]] is not None and all(grid[cell] == grid[line[0]] for cell in line)}
            if winners:
                (winner,) = winners
                expected = (1.0, 0.0) if winner == "X" else (0.0, 1.0)
            elif None not in marks:
                expected = (0.5, 0.5)
            else:
                expected = (None, None)
            assert (values["payoff(X)"], values["payoff(O)"]) == expected
            assert marks.count("X") - marks.count("O") in (0, 1)
            if variant.gravity:
                assert all(
                    mark is None or row == variant.height or grid[(row + 1, col)] is not None
                    for (row, col), mark in grid.items()
                )
        assert dict(state.variables)["payoff(X)"] is not None
