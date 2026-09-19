from collections.abc import Callable
from pathlib import Path

import pytest

from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.model.sudoku_puzzle import SudokuPuzzle
from openmind.agent.factory.sudoku_factory import declare_sudoku
from openmind.agent.repository.sudoku_puzzle_repository import SudokuPuzzleRepository
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_game
from openmind.rbs.service.rule_based_game import RuleBasedGame

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedGame]

DIRECTORY = Path(__file__).parents[2] / "data" / "sudoku"
REPOSITORY = SudokuPuzzleRepository(SudokuCollectionMapper())
PUZZLES = [
    puzzle for collection in REPOSITORY.collections(DIRECTORY) for puzzle in REPOSITORY.load(DIRECTORY, collection)
]
POSITIONS = range(9)
UNITS = (
    [[(row, col) for col in POSITIONS] for row in POSITIONS]
    + [[(row, col) for row in POSITIONS] for col in POSITIONS]
    + [[(top + row, left + col) for row in range(3) for col in range(3)] for top in (0, 3, 6) for left in (0, 3, 6)]
)


@pytest.mark.skipif(not PUZZLES, reason="no collection in data/sudoku; the README's Solve section has the downloads")
@pytest.mark.parametrize("puzzle", PUZZLES, ids=lambda puzzle: f"{puzzle.collection}/{puzzle.number}")
def test_every_published_puzzle_has_one_solution_that_keeps_its_clues_and_follows_the_rules(
    knowledge: KnowledgeBase, puzzle: SudokuPuzzle
) -> None:
    context = declare_sudoku(knowledge, f"sudoku/{puzzle.collection}/{puzzle.number}", puzzle.grid)
    rbs = create_rule_based_game(knowledge, context)

    actions = rbs.actions(rbs.start(), limit=2)

    assert len(actions) == 1
    ((outcome, probability),) = rbs.outcomes(rbs.start(), actions[0]).outcomes
    cell = outcome.model("cell")
    grid = [[cell[row + 1, col + 1] for col in POSITIONS] for row in POSITIONS]
    assert all(sorted(grid[row][col] for row, col in unit) == list(range(1, 10)) for unit in UNITS)
    assert all(mark == "." or grid[index // 9][index % 9] == int(mark) for index, mark in enumerate(puzzle.grid))
    assert (probability, outcome.model("payoff")["solver"]) == (1.0, 1.0)
