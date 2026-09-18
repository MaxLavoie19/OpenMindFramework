import textwrap

from openmind.agent.constant.rock_paper_scissors_constant import (
    BEATS,
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
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.model.python_rule import PythonRule
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


def declare_rock_paper_scissors_moves(declarer: RuleDeclarer) -> None:
    """A player to act who hasn't thrown throws a shape."""
    declarer.values(THROW, SHAPE, PythonRule(repr(SHAPES)))
    declarer.constraints(
        THROW, PythonRule(f"{TURN}[{PLAYER}]"), PythonRule(f"{HAND}[{PLAYER}] is {UNSET!r}")
    )


def declare_rock_paper_scissors_effects(declarer: RuleDeclarer) -> None:
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
    declarer.definitions(create_rock_paper_scissors_definitions(), effects=True)
    declarer.leads_to(THROW, PythonRule(f"{HAND}[{PLAYER}] = {SHAPE}"))
    declarer.together(PythonRule(resolution))


def create_rock_paper_scissors_players() -> Players:
    """A and B; turn(A) and turn(B) flag the players to act; payoff(A) and payoff(B) hold their payoffs."""
    variable_name_mapper = VariableNameMapper()
    return Players(PLAYERS, TURN, tuple(variable_name_mapper.to_name(PAYOFF, (player,)) for player in PLAYERS))


def declare_rock_paper_scissors(knowledge_base: KnowledgeBase, weight: float = 1.0) -> str:
    """Declares rock paper scissors' rules and gives back the context they were declared under: A and B throw at once;
    rock beats scissors, paper beats rock, scissors beat paper."""
    declarer = RuleDeclarer(knowledge_base, NAME, weight)
    declarer.starts_at(create_rock_paper_scissors_initial_state())
    declarer.played_by(create_rock_paper_scissors_players())
    declarer.definitions(create_rock_paper_scissors_definitions())
    declare_rock_paper_scissors_moves(declarer)
    declare_rock_paper_scissors_effects(declarer)
    return declarer.done()
