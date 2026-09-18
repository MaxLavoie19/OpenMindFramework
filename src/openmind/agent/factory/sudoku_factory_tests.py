from collections.abc import Callable

import pytest

from openmind.agent.factory.sudoku_factory import (
    create_sudoku_initial_state,
    create_sudoku_players,
    declare_sudoku,
)
from openmind.doxastic.constant.rule_kind_constant import CONSTRAINT, EFFECTS, VALUES
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.model.python_rule import PythonRule
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.players import Players

TOP95_FIRST = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"

type Game = Callable[[str], RuleBasedSystem]


def test_initial_state_holds_the_clues_and_the_empty_cells() -> None:
    variables = dict(create_sudoku_initial_state().variables)

    assert (variables["cell(1,1)"], variables["cell(1,2)"], variables["cell(1,3)"], variables["cell(9,9)"]) == (
        5,
        3,
        None,
        9,
    )
    assert sum(1 for name, value in variables.items() if name.startswith("cell") and value is None) == 51
    assert (variables["turn"], variables["payoff"], len(variables)) == ("solver", None, 83)


def test_every_empty_cell_is_a_parameter_under_one_all_different_per_row_column_and_box(
    knowledge: KnowledgeBase,
) -> None:
    context = declare_sudoku(knowledge)

    values, constraints = knowledge.rules(context, (VALUES,)), knowledge.rules(context, (CONSTRAINT,))

    assert (len(values), len(constraints)) == (51, 28)
    assert values[0].parameter == "cell_1_3"
    assert values[0].rule == PythonRule("(1, 2, 3, 4, 5, 6, 7, 8, 9)")
    assert constraints[0].rule == PythonRule("payoff is None")
    assert constraints[1].rule == PythonRule(
        "all_different(cell[1, 1], cell[1, 2], cell_1_3, cell_1_4, cell[1, 5], cell_1_6, cell_1_7, cell_1_8, cell_1_9)"
    )


def test_filling_writes_every_parameter_into_its_cell_and_pays_one(knowledge: KnowledgeBase) -> None:
    context = declare_sudoku(knowledge)

    (effects,) = knowledge.rules(context, (EFFECTS,))
    lines = effects.rule.source.splitlines()

    assert (effects.action, effects.probability, len(lines)) == ("fill", 1.0, 52)
    assert (lines[0], lines[-1]) == ("cell[1, 3] = cell_1_3", "payoff = 1.0")


def test_players_are_a_single_solver() -> None:
    assert create_sudoku_players() == Players(("solver",), "turn", ("payoff",))


def test_the_declared_game_starts_where_sudoku_starts(game: Game) -> None:
    rbs = game("sudoku")

    assert rbs.context == "sudoku"
    assert rbs.start() == create_sudoku_initial_state()
    assert rbs.empties() == (("cell", None),)


def test_a_name_and_a_grid_declare_that_puzzle_under_its_own_context(knowledge: KnowledgeBase) -> None:
    context = declare_sudoku(knowledge, "sudoku/top95/1", TOP95_FIRST)
    variables = dict(create_sudoku_initial_state(TOP95_FIRST).variables)
    (effects,) = knowledge.rules(context, (EFFECTS,))

    assert context == "sudoku/top95/1"
    assert (variables["cell(1,1)"], variables["cell(1,2)"], variables["cell(1,7)"]) == (4, None, 8)
    assert (len(knowledge.rules(context, (VALUES,))), len(effects.rule.source.splitlines())) == (64, 65)


@pytest.mark.parametrize("grid", [TOP95_FIRST[:80], TOP95_FIRST + ".", TOP95_FIRST.replace(".", "0", 1)])
def test_a_grid_that_is_not_81_empty_marks_or_digits_raises(grid: str) -> None:
    with pytest.raises(ValueError, match="A sudoku grid needs 81 characters"):
        create_sudoku_initial_state(grid)
