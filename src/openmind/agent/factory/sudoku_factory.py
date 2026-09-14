from openmind.agent.builder.domain_builder import DomainBuilder
from openmind.agent.constant.sudoku_constant import (
    BOX,
    CELL,
    CELLS,
    CERTAIN,
    CLUE_MARKS,
    DIGITS,
    EMPTY,
    EMPTY_MARK,
    FILL,
    NAME,
    PARAMETER,
    PAYOFF,
    PLAYER,
    PUZZLE,
    SIZE,
    SOLVED,
    TURN,
    UNSET,
)
from openmind.agent.model.domain import Domain
from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.constant.rule_constant import ALL_DIFFERENT
from openmind.rule.model.python_rule import PythonRule
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.model.value import Value


def create_sudoku_initial_state(grid: str = PUZZLE) -> State:
    """The grid's clues in their cells, every other cell empty, the solver to act, no payoff yet."""
    names = VariableNameMapper()
    builder = StateBuilder()
    for row, col, clue in _cells(grid):
        builder.with_variable(names.to_name(CELL, (row, col)), clue)
    return builder.with_variable(TURN, PLAYER).with_variable(PAYOFF, UNSET).build()


def create_sudoku_problem(grid: str = PUZZLE) -> Problem:
    """Fill every empty cell of the grid at once, with no digit twice in a row, a column or a box. Each empty cell is a
    parameter named like cell_1_3; a clue is read from the state as cell[1, 1]."""
    clues = {(row, col): clue for row, col, clue in _cells(grid)}
    digits = DiscreteDomain(DIGITS)
    constraints = (
        PythonRule(f"{PAYOFF} is {UNSET!r}"),
        *(
            PythonRule(f"{ALL_DIFFERENT}({', '.join(_operand(row, col, clues[row, col]) for row, col in unit)})")
            for unit in _units()
        ),
    )
    variables = tuple(Variable(_parameter(row, col), digits) for (row, col), clue in clues.items() if clue is EMPTY)
    return ProblemBuilder().with_action(FILL, variables, constraints).build()


def create_sudoku_transitions(grid: str = PUZZLE) -> TransitionModel:
    """Write every filled value into its cell of the grid; a full grid pays 1.0, the share of cells filled."""
    effects = "\n".join(
        (
            *(f"{CELL}[{row}, {col}] = {_parameter(row, col)}" for row, col, clue in _cells(grid) if clue is EMPTY),
            f"{PAYOFF} = {SOLVED!r}",
        )
    )
    return TransitionModelBuilder().with_transition(FILL, (Branch(CERTAIN, PythonRule(effects)),)).build()


def create_sudoku_players() -> Players:
    """A single player, the solver: turn names it and payoff holds its payoff."""
    return Players((PLAYER,), TURN, (PAYOFF,))


def create_sudoku_domain(name: str = NAME, grid: str = PUZZLE) -> Domain:
    """Sudoku: the grid, the constraints that fill it, the transition that writes the solution, and the solver. Without
    arguments, the domain "sudoku" holds the puzzle written here."""
    return (
        DomainBuilder()
        .with_name(name)
        .with_initial_state(create_sudoku_initial_state(grid))
        .with_problem(create_sudoku_problem(grid))
        .with_transitions(create_sudoku_transitions(grid))
        .with_players(create_sudoku_players())
        .build()
    )


def _cells(grid: str) -> list[tuple[int, int, Value]]:
    """Every cell's row, column and clue, or EMPTY, row by row; a grid that isn't 81 empty marks or digits raises
    ValueError."""
    if len(grid) != CELLS or any(mark != EMPTY_MARK and mark not in CLUE_MARKS for mark in grid):
        raise ValueError(
            f"A sudoku grid needs {CELLS} characters, each {EMPTY_MARK!r} or a digit from 1 to {SIZE}: {grid!r}"
        )
    return [
        (index // SIZE + 1, index % SIZE + 1, EMPTY if mark == EMPTY_MARK else int(mark))
        for index, mark in enumerate(grid)
    ]


def _units() -> list[list[tuple[int, int]]]:
    """The cells of every row, column and box."""
    positions = range(1, SIZE + 1)
    rows = [[(row, col) for col in positions] for row in positions]
    columns = [[(row, col) for row in positions] for col in positions]
    boxes = [
        [(top + row, left + col) for row in range(1, BOX + 1) for col in range(1, BOX + 1)]
        for top in range(0, SIZE, BOX)
        for left in range(0, SIZE, BOX)
    ]
    return rows + columns + boxes


def _parameter(row: int, col: int) -> str:
    return PARAMETER.format(row=row, col=col)


def _operand(row: int, col: int, clue: Value) -> str:
    return _parameter(row, col) if clue is EMPTY else f"{CELL}[{row}, {col}]"
