from openmind.agent.constant.sudoku_constant import (
    BOX,
    CELL,
    CELLS,
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
from openmind.rbs.constant.rule_constant import ALL_DIFFERENT
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.structure.model.value import Value


def create_sudoku_initial_state(grid: str = PUZZLE) -> State:
    """The cell grid, 9 by 9, holding the puzzle's clues in their cells, every other cell empty; the solver to act; the
    payoff map holding no payoff yet."""
    return (
        StateBuilder()
        .with_model(CELL, Grid((SIZE, SIZE), tuple(clue for _, _, clue in _cells(grid))))
        .with_model(TURN, PLAYER)
        .with_model(PAYOFF, Map.of({PLAYER: UNSET}))
        .build()
    )


def declare_sudoku_moves(declarer: RuleDeclarer, grid: str = PUZZLE) -> None:
    """Fill every empty cell of the grid at once, with no digit twice in a row, a column or a box. Each empty cell is a
    parameter named like cell_1_3; a clue is read from the state as cell[1, 1]."""
    clues = {(row, col): clue for row, col, clue in _cells(grid)}
    digits = PythonRule(repr(DIGITS))
    for (row, col), clue in clues.items():
        if clue is EMPTY:
            declarer.values(FILL, _parameter(row, col), digits)
    declarer.constraints(
        FILL,
        PythonRule(f"{PAYOFF}[{PLAYER!r}] is {UNSET!r}"),
        *(
            PythonRule(f"{ALL_DIFFERENT}({', '.join(_operand(row, col, clues[row, col]) for row, col in unit)})")
            for unit in _units()
        ),
    )


def declare_sudoku_effects(declarer: RuleDeclarer, grid: str = PUZZLE) -> None:
    """Place every filled value in its cell of the grid; a full grid pays the solver 1.0, the share of cells filled."""
    effects = "\n".join(
        (
            *(
                f"{CELL} = {CELL}.placed(({row}, {col}), {_parameter(row, col)})"
                for row, col, clue in _cells(grid)
                if clue is EMPTY
            ),
            f"{PAYOFF} = {PAYOFF}.with_item({PLAYER!r}, {SOLVED!r})",
        )
    )
    declarer.leads_to(FILL, PythonRule(effects))


def create_sudoku_players() -> Players:
    """A single player, the solver: turn names it and the payoff map holds its payoff."""
    return Players((PLAYER,), TURN, PAYOFF)


def declare_sudoku(
    knowledge_base: KnowledgeBase, name: str = NAME, grid: str = PUZZLE, weight: float = 1.0
) -> str:
    """Declares sudoku's rules into the knowledge base and gives back the context they were declared under. Without
    arguments, the context "sudoku" holds the puzzle written here."""
    declarer = RuleDeclarer(knowledge_base, name, weight)
    declarer.starts_at(create_sudoku_initial_state(grid))
    declarer.played_by(create_sudoku_players())
    declarer.empty(CELL, EMPTY)
    declare_sudoku_moves(declarer, grid)
    declare_sudoku_effects(declarer, grid)
    return declarer.done()


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


def _units() -> list[tuple[tuple[int, ...], ...]]:
    """The cells of every row, column and box of the grid."""
    board = Grid.filled((SIZE, SIZE), EMPTY)
    return [*board.rows(), *board.columns(), *board.boxes((BOX, BOX))]


def _parameter(row: int, col: int) -> str:
    return PARAMETER.format(row=row, col=col)


def _operand(row: int, col: int, clue: Value) -> str:
    return _parameter(row, col) if clue is EMPTY else f"{CELL}[{row}, {col}]"
