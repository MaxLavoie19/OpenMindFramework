from openmind.agent.constant.tictactoe_constant import (
    BOARD_SIZE,
    CELL,
    CERTAIN,
    COL,
    DRAW,
    EMPTY,
    LOSS,
    PAYOFF,
    PLACE,
    PLAYERS,
    ROW,
    TURN,
    UNSET,
    WIN,
)
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
from openmind.predictor.model.transition_model import TransitionModel
from openmind.predictor.model.when import When
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State


def create_tictactoe_initial_state() -> State:
    """Every cell empty, X to play, no payoff set."""
    variable_name_mapper = VariableNameMapper()
    builder = StateBuilder()
    for row in range(1, BOARD_SIZE + 1):
        for col in range(1, BOARD_SIZE + 1):
            builder.with_variable(variable_name_mapper.to_name(CELL, (row, col)), EMPTY)
    builder.with_variable(TURN, PLAYERS[0])
    for player in PLAYERS:
        builder.with_variable(variable_name_mapper.to_name(PAYOFF, (player,)), UNSET)
    return builder.build()


def create_tictactoe_problem() -> Problem:
    """Place a mark on an empty cell, as long as no payoff is set."""
    positions = DiscreteDomain(tuple(range(1, BOARD_SIZE + 1)))
    no_payoff_set = tuple(
        Equals(StateVariable(PAYOFF, (Constant(player),)), Constant(UNSET)) for player in PLAYERS
    )
    cell_is_empty = Equals(
        StateVariable(CELL, (ActionParameter(ROW), ActionParameter(COL))), Constant(EMPTY)
    )
    return (
        ProblemBuilder()
        .with_action(
            PLACE,
            (Variable(ROW, positions), Variable(COL, positions)),
            (*no_payoff_set, cell_is_empty),
        )
        .build()
    )


def create_tictactoe_transitions() -> TransitionModel:
    """Mark the cell, set payoffs on a win or a full board, then pass the turn."""
    positions = range(1, BOARD_SIZE + 1)
    rows = [[(row, col) for col in positions] for row in positions]
    columns = [[(row, col) for row in positions] for col in positions]
    diagonals = [[(i, i) for i in positions], [(i, BOARD_SIZE + 1 - i) for i in positions]]
    cells = [cell for row in rows for cell in row]
    first, second = PLAYERS

    mark_cell = Assign(
        StateVariable(CELL, (ActionParameter(ROW), ActionParameter(COL))), StateVariable(TURN)
    )
    wins = tuple(
        When(
            AnyOf(
                tuple(
                    AllOf(
                        tuple(
                            Equals(StateVariable(CELL, (Constant(row), Constant(col))), Constant(player))
                            for row, col in line
                        )
                    )
                    for line in rows + columns + diagonals
                )
            ),
            (
                Assign(StateVariable(PAYOFF, (Constant(player),)), Constant(WIN)),
                Assign(StateVariable(PAYOFF, (Constant(opponent),)), Constant(LOSS)),
            ),
        )
        for player, opponent in ((first, second), (second, first))
    )
    draw = When(
        AllOf(
            (
                Equals(StateVariable(PAYOFF, (Constant(first),)), Constant(UNSET)),
                *(
                    Not(Equals(StateVariable(CELL, (Constant(row), Constant(col))), Constant(EMPTY)))
                    for row, col in cells
                ),
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
        .with_transition(PLACE, (Branch(CERTAIN, (mark_cell, *wins, draw, pass_turn)),))
        .build()
    )
