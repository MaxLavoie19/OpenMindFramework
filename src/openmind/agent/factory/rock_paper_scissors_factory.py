import textwrap

from openmind.agent.builder.domain_builder import DomainBuilder
from openmind.agent.constant.rock_paper_scissors_constant import (
    BEATS,
    CERTAIN,
    DRAW,
    HAND,
    LOSS,
    NAME,
    PAYOFF,
    PLAYERS,
    SHAPE,
    SHAPES,
    THROW,
    TURN,
    UNSET,
    WIN,
)
from openmind.agent.model.domain import Domain
from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.predictor.builder.transition_model_builder import TransitionModelBuilder
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.constant.players_constant import PLAYER
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.players import Players
from openmind.world.model.state import State


def create_rock_paper_scissors_initial_state() -> State:
    """Both players to act at once, turn(A) and turn(B) true; no hand thrown, no payoff set."""
    variable_name_mapper = VariableNameMapper()
    builder = StateBuilder()
    for player in PLAYERS:
        builder.with_variable(variable_name_mapper.to_name(HAND, (player,)), UNSET)
        builder.with_variable(variable_name_mapper.to_name(PAYOFF, (player,)), UNSET)
        builder.with_variable(variable_name_mapper.to_name(TURN, (player,)), True)
    return builder.build()


def create_rock_paper_scissors_definitions() -> PythonRule:
    """The names every rule sees: the PLAYERS, the SHAPES, BEATS, each shape and the shape it beats, and the payoffs WIN,
    DRAW and LOSS."""
    return PythonRule(
        textwrap.dedent(
            f"""\
            PLAYERS = {PLAYERS!r}
            SHAPES = {SHAPES!r}
            BEATS = {BEATS!r}
            WIN, DRAW, LOSS = {WIN!r}, {DRAW!r}, {LOSS!r}
            """
        )
    )


def create_rock_paper_scissors_problem() -> Problem:
    """A player to act who hasn't thrown throws a shape."""
    return (
        ProblemBuilder()
        .with_action(
            THROW,
            (Variable(SHAPE, DiscreteDomain(SHAPES)),),
            (PythonRule(f"{TURN}[{PLAYER}]"), PythonRule(f"{HAND}[{PLAYER}] is {UNSET!r}")),
        )
        .build()
    )


def create_rock_paper_scissors_transitions() -> TransitionModel:
    """Each throw sets its player's hand; once both have thrown, the resolution compares the hands, sets the payoffs and
    ends the game."""
    resolution = textwrap.dedent(
        f"""\
        first, second = {HAND}[PLAYERS[0]], {HAND}[PLAYERS[1]]
        if first == second:
            {PAYOFF}[PLAYERS[0]], {PAYOFF}[PLAYERS[1]] = DRAW, DRAW
        elif BEATS[first] == second:
            {PAYOFF}[PLAYERS[0]], {PAYOFF}[PLAYERS[1]] = WIN, LOSS
        else:
            {PAYOFF}[PLAYERS[0]], {PAYOFF}[PLAYERS[1]] = LOSS, WIN
        for thrower in PLAYERS:
            {TURN}[thrower] = False
        """
    )
    return (
        TransitionModelBuilder()
        .with_definitions(create_rock_paper_scissors_definitions())
        .with_transition(THROW, (Branch(CERTAIN, PythonRule(f"{HAND}[{PLAYER}] = {SHAPE}")),))
        .with_resolution((Branch(CERTAIN, PythonRule(resolution)),))
        .build()
    )


def create_rock_paper_scissors_players() -> Players:
    """A and B; turn(A) and turn(B) flag the players to act; payoff(A) and payoff(B) hold their payoffs."""
    variable_name_mapper = VariableNameMapper()
    return Players(PLAYERS, TURN, tuple(variable_name_mapper.to_name(PAYOFF, (player,)) for player in PLAYERS))


def create_rock_paper_scissors_domain() -> Domain:
    """Rock paper scissors: A and B throw at once; rock beats scissors, paper beats rock, scissors beat paper."""
    return (
        DomainBuilder()
        .with_name(NAME)
        .with_initial_state(create_rock_paper_scissors_initial_state())
        .with_problem(create_rock_paper_scissors_problem())
        .with_transitions(create_rock_paper_scissors_transitions())
        .with_players(create_rock_paper_scissors_players())
        .build()
    )
