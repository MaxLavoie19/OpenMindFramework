from openmind.agent.constant.tictactoe_constant import (
    BOARD_SIZE,
    CELL,
    COL,
    EMPTY,
    PAYOFF,
    PLACE,
    PLAYERS,
    ROW,
    TURN,
    UNSET,
)
from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
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
