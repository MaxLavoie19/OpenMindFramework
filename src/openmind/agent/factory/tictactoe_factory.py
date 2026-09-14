import textwrap

from openmind.agent.builder.domain_builder import DomainBuilder
from openmind.agent.constant.tictactoe_constant import (
    CELL,
    CERTAIN,
    COL,
    DIRECTIONS,
    DRAW,
    DROP,
    EMPTY,
    LOSS,
    NAME,
    PAYOFF,
    PLACE,
    PLAYERS,
    ROW,
    SEPARATOR,
    STANDARD,
    TURN,
    UNSET,
    WIN,
)
from openmind.agent.model.domain import Domain
from openmind.agent.model.tictactoe_variant import TicTacToeVariant
from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State


def create_tictactoe_initial_state(variant: TicTacToeVariant = STANDARD) -> State:
    """Every cell of the variant's grid empty, row 1 at the top; X to play; no payoff set."""
    _check(variant)
    variable_name_mapper = VariableNameMapper()
    builder = StateBuilder()
    for row in range(1, variant.height + 1):
        for col in range(1, variant.width + 1):
            builder.with_variable(variable_name_mapper.to_name(CELL, (row, col)), EMPTY)
    builder.with_variable(TURN, PLAYERS[0])
    for player in PLAYERS:
        builder.with_variable(variable_name_mapper.to_name(PAYOFF, (player,)), UNSET)
    return builder.build()


def create_tictactoe_definitions(variant: TicTacToeVariant = STANDARD) -> PythonRule:
    """The names every tic-tac-toe rule sees: the variant's WIDTH, HEIGHT and LINE, the PLAYERS, the WIN, DRAW and LOSS
    payoffs, other(player), and LINES_THROUGH[row, col], every line of LINE cells through a cell that fits the grid."""
    _check(variant)
    return PythonRule(
        textwrap.dedent(
            f"""\
            WIDTH, HEIGHT, LINE = {variant.width}, {variant.height}, {variant.line}
            PLAYERS = {PLAYERS!r}
            WIN, DRAW, LOSS = {WIN!r}, {DRAW!r}, {LOSS!r}


            def other(player):
                return PLAYERS[1] if player == PLAYERS[0] else PLAYERS[0]


            def lines_through(row, col):
                lines = []
                for row_step, col_step in {DIRECTIONS!r}:
                    for offset in range(LINE):
                        line = [(row + (index - offset) * row_step, col + (index - offset) * col_step) for index in range(LINE)]
                        if all(1 <= r <= HEIGHT and 1 <= c <= WIDTH for r, c in line) and line not in lines:
                            lines.append(line)
                return lines


            LINES_THROUGH = {{(row, col): lines_through(row, col) for row in range(1, HEIGHT + 1) for col in range(1, WIDTH + 1)}}
            """
        )
    )


def create_tictactoe_problem(variant: TicTacToeVariant = STANDARD) -> Problem:
    """While no payoff is set: place a mark on an empty cell or, with gravity, drop it in a column whose top cell is
    empty."""
    _check(variant)
    no_payoff_set = tuple(PythonRule(f"{PAYOFF}[{player!r}] is {UNSET!r}") for player in PLAYERS)
    columns = DiscreteDomain(tuple(range(1, variant.width + 1)))
    builder = ProblemBuilder().with_definitions(create_tictactoe_definitions(variant))
    if variant.gravity:
        top_cell_is_empty = PythonRule(f"{CELL}[1, {COL}] is {EMPTY!r}")
        return builder.with_action(DROP, (Variable(COL, columns),), (*no_payoff_set, top_cell_is_empty)).build()
    rows = DiscreteDomain(tuple(range(1, variant.height + 1)))
    cell_is_empty = PythonRule(f"{CELL}[{ROW}, {COL}] is {EMPTY!r}")
    return builder.with_action(
        PLACE, (Variable(ROW, rows), Variable(COL, columns)), (*no_payoff_set, cell_is_empty)
    ).build()


def create_tictactoe_transitions(variant: TicTacToeVariant = STANDARD) -> TransitionModel:
    """Mark the landing cell and check only the lines through it for a win, set a draw on a full board, then pass the
    turn."""
    _check(variant)
    first, second = PLAYERS
    if variant.gravity:
        action = DROP
        landing = f"{ROW} = max(r for r in range(1, HEIGHT + 1) if {CELL}[r, {COL}] is {EMPTY!r})\n"
        full = f"all({CELL}[1, c] is not {EMPTY!r} for c in range(1, WIDTH + 1))"
    else:
        action = PLACE
        landing = ""
        full = f"all(mark is not {EMPTY!r} for mark in {CELL}.values())"
    effects = landing + textwrap.dedent(
        f"""\
        {CELL}[{ROW}, {COL}] = {TURN}
        if any(all({CELL}[r, c] == {TURN} for r, c in line) for line in LINES_THROUGH[{ROW}, {COL}]):
            {PAYOFF}[{TURN}] = WIN
            {PAYOFF}[other({TURN})] = LOSS
        elif {full}:
            {PAYOFF}[{first!r}] = DRAW
            {PAYOFF}[{second!r}] = DRAW
        {TURN} = other({TURN})
        """
    )
    return (
        TransitionModelBuilder()
        .with_definitions(create_tictactoe_definitions(variant))
        .with_transition(action, (Branch(CERTAIN, PythonRule(effects)),))
        .build()
    )


def create_tictactoe_players() -> Players:
    """X and O; turn names the player to act; payoff(X) and payoff(O) hold their payoffs."""
    variable_name_mapper = VariableNameMapper()
    return Players(
        PLAYERS, TURN, tuple(variable_name_mapper.to_name(PAYOFF, (player,)) for player in PLAYERS)
    )


def create_tictactoe_domain(variant: TicTacToeVariant = STANDARD) -> Domain:
    """Tic-tac-toe or one of its variants: its initial state, constraints, transitions and players. The standard game is
    named "tictactoe", any other variant "tictactoe/<variant>"."""
    return (
        DomainBuilder()
        .with_name(NAME if variant == STANDARD else SEPARATOR.join((NAME, variant.name)))
        .with_initial_state(create_tictactoe_initial_state(variant))
        .with_problem(create_tictactoe_problem(variant))
        .with_transitions(create_tictactoe_transitions(variant))
        .with_players(create_tictactoe_players())
        .build()
    )


def _check(variant: TicTacToeVariant) -> None:
    if variant.width < 1 or variant.height < 1 or not 1 <= variant.line <= max(variant.width, variant.height):
        raise ValueError(
            "A tic-tac-toe variant needs a width and a height of at least 1 and a line from 1 to its longer side, "
            f"not {variant!r}"
        )
