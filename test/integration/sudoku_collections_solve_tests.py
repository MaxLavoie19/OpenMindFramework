from pathlib import Path

import pytest

from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.model.sudoku_puzzle import SudokuPuzzle
from openmind.agent.repository.sudoku_puzzle_repository import SudokuPuzzleRepository
from openmind.csp.factory.csp_factory import create_solver
from openmind.predictor.factory.predictor_factory import create_predictor

pytestmark = pytest.mark.log_level("INFO")

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
    puzzle: SudokuPuzzle,
) -> None:
    domain = create_sudoku_domain(f"sudoku/{puzzle.collection}/{puzzle.number}", puzzle.grid)
    predictor = create_predictor()

    actions = create_solver().solve(domain.problem, domain.initial_state, limit=2)

    assert len(actions) == 1
    ((outcome, probability),) = predictor.predict(domain.transitions, domain.initial_state, actions[0]).outcomes
    values = dict(outcome.variables)
    grid = [[values[f"cell({row + 1},{col + 1})"] for col in POSITIONS] for row in POSITIONS]
    assert all(sorted(grid[row][col] for row, col in unit) == list(range(1, 10)) for unit in UNITS)
    assert all(mark == "." or grid[index // 9][index % 9] == int(mark) for index, mark in enumerate(puzzle.grid))
    assert (probability, values["payoff"]) == (1.0, 1.0)
