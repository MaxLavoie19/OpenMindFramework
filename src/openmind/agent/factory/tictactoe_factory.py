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
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.all_of import AllOf
from openmind.expression.model.any_of import AnyOf
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.not_ import Not
from openmind.expression.model.state_variable import StateVariable
from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.assign import Assign
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.effect import Effect
from openmind.predictor.model.transition_model import TransitionModel
from openmind.predictor.model.when import When
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.model.value import Value


def create_tictactoe_initial_state(variant: TicTacToeVariant = STANDARD) -> State:
    """Every cell of the variant's grid empty, row 1 at the top; X to play; no payoff set."""
    _check(variant)
    variable_name_mapper = VariableNameMapper()
    builder = StateBuilder()
    for row, col in _cells(variant):
        builder.with_variable(variable_name_mapper.to_name(CELL, (row, col)), EMPTY)
    builder.with_variable(TURN, PLAYERS[0])
    for player in PLAYERS:
        builder.with_variable(variable_name_mapper.to_name(PAYOFF, (player,)), UNSET)
    return builder.build()


def create_tictactoe_problem(variant: TicTacToeVariant = STANDARD) -> Problem:
    """While no payoff is set: place a mark on an empty cell or, with gravity, drop it in a column whose top cell is
    empty."""
    _check(variant)
    no_payoff_set = tuple(_payoff_is(player, UNSET) for player in PLAYERS)
    columns = DiscreteDomain(tuple(range(1, variant.width + 1)))
    if variant.gravity:
        top_cell_is_empty = Equals(StateVariable(CELL, (Constant(1), ActionParameter(COL))), Constant(EMPTY))
        return (
            ProblemBuilder()
            .with_action(DROP, (Variable(COL, columns),), (*no_payoff_set, top_cell_is_empty))
            .build()
        )
    rows = DiscreteDomain(tuple(range(1, variant.height + 1)))
    cell_is_empty = Equals(StateVariable(CELL, (ActionParameter(ROW), ActionParameter(COL))), Constant(EMPTY))
    return (
        ProblemBuilder()
        .with_action(PLACE, (Variable(ROW, rows), Variable(COL, columns)), (*no_payoff_set, cell_is_empty))
        .build()
    )


def create_tictactoe_transitions(variant: TicTacToeVariant = STANDARD) -> TransitionModel:
    """Mark the landing cell and check only the lines through it for a win, set a draw on a full board, then pass the
    turn."""
    _check(variant)
    first, second = PLAYERS
    rows, columns = range(1, variant.height + 1), range(1, variant.width + 1)
    if variant.gravity:
        action = DROP
        mark: tuple[Effect, ...] = tuple(
            When(Equals(ActionParameter(COL), Constant(col)), (_fall(variant, variant.height, col),)) for col in columns
        )
        cells_filled_last = [(1, col) for col in columns]
    else:
        action = PLACE
        mark = tuple(
            When(
                Equals(ActionParameter(ROW), Constant(row)),
                tuple(When(Equals(ActionParameter(COL), Constant(col)), _land(variant, row, col)) for col in columns),
            )
            for row in rows
        )
        cells_filled_last = _cells(variant)
    draw = When(
        AllOf(
            (
                _payoff_is(first, UNSET),
                *(Not(Equals(_cell(row, col), Constant(EMPTY))) for row, col in cells_filled_last),
            )
        ),
        tuple(Assign(StateVariable(PAYOFF, (Constant(player),)), Constant(DRAW)) for player in PLAYERS),
    )
    pass_turn = When(
        Equals(StateVariable(TURN), Constant(first)),
        (Assign(StateVariable(TURN), Constant(second)),),
        (Assign(StateVariable(TURN), Constant(first)),),
    )
    return (
        TransitionModelBuilder()
        .with_transition(action, (Branch(CERTAIN, (*mark, draw, pass_turn)),))
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


def _fall(variant: TicTacToeVariant, row: int, col: int) -> When:
    """With gravity, the mark lands in (row, col) when it is empty, otherwise it tries the cell above."""
    return When(
        Equals(_cell(row, col), Constant(EMPTY)),
        _land(variant, row, col),
        (_fall(variant, row - 1, col),) if row > 1 else (),
    )


def _land(variant: TicTacToeVariant, row: int, col: int) -> tuple[Effect, ...]:
    """Marks (row, col) for the player to act; when the other cells of a line through it hold that player's marks, the
    player wins and the other player loses."""
    first, second = PLAYERS
    turn = StateVariable(TURN)
    completes_a_line = AnyOf(
        tuple(
            AllOf(tuple(Equals(_cell(line_row, line_col), turn) for line_row, line_col in line if (line_row, line_col) != (row, col)))
            for line in _lines_through(variant, row, col)
        )
    )
    win = When(
        completes_a_line,
        (
            Assign(StateVariable(PAYOFF, (turn,)), Constant(WIN)),
            When(
                Equals(turn, Constant(first)),
                (Assign(StateVariable(PAYOFF, (Constant(second),)), Constant(LOSS)),),
                (Assign(StateVariable(PAYOFF, (Constant(first),)), Constant(LOSS)),),
            ),
        ),
    )
    return (Assign(_cell(row, col), turn), win)


def _lines_through(variant: TicTacToeVariant, row: int, col: int) -> list[list[tuple[int, int]]]:
    """Every line of variant.line cells through (row, col), in any of the four directions, that fits in the grid."""
    lines: list[list[tuple[int, int]]] = []
    for row_step, col_step in DIRECTIONS:
        for offset in range(variant.line):
            line = [
                (row + (index - offset) * row_step, col + (index - offset) * col_step) for index in range(variant.line)
            ]
            fits = all(1 <= line_row <= variant.height and 1 <= line_col <= variant.width for line_row, line_col in line)
            if fits and line not in lines:
                lines.append(line)
    return lines


def _cells(variant: TicTacToeVariant) -> list[tuple[int, int]]:
    return [(row, col) for row in range(1, variant.height + 1) for col in range(1, variant.width + 1)]


def _cell(row: int, col: int) -> StateVariable:
    return StateVariable(CELL, (Constant(row), Constant(col)))


def _payoff_is(player: str, value: Value) -> Equals:
    return Equals(StateVariable(PAYOFF, (Constant(player),)), Constant(value))


def _check(variant: TicTacToeVariant) -> None:
    if variant.width < 1 or variant.height < 1 or not 1 <= variant.line <= max(variant.width, variant.height):
        raise ValueError(
            "A tic-tac-toe variant needs a width and a height of at least 1 and a line from 1 to its longer side, "
            f"not {variant!r}"
        )
