from collections.abc import Callable
import pytest


pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]

SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def test_the_single_solution_fills_the_known_grid_and_pays_one(game: Game) -> None:
    rbs = game("sudoku")

    actions = rbs.actions(rbs.start(), limit=2)

    assert len(actions) == 1
    ((outcome, probability),) = rbs.outcomes(rbs.start(), actions[0]).outcomes
    values = dict(outcome.variables)
    assert "".join(str(values[f"cell({row},{col})"]) for row in range(1, 10) for col in range(1, 10)) == SOLUTION
    assert (probability, values["payoff"]) == (1.0, 1.0)


def test_searching_without_a_limit_finds_no_other_solution(game: Game) -> None:
    rbs = game("sudoku")

    assert len(rbs.actions(rbs.start())) == 1
