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
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.map import Map
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.constant.players_constant import PLAYER
from openmind.world.model.players import Players
from openmind.world.model.state import State


def create_rock_paper_scissors_initial_state() -> State:
    """Both players to act at once, the turn map flagging A and B true; the hand map holding no hand thrown; the payoff
    map holding no payoff."""
    return (
        StateBuilder()
        .with_model(HAND, Map.of(dict.fromkeys(PLAYERS, UNSET)))
        .with_model(PAYOFF, Map.of(dict.fromkeys(PLAYERS, UNSET)))
        .with_model(TURN, Map.of(dict.fromkeys(PLAYERS, True)))
        .build()
    )


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
            {PAYOFF} = {PAYOFF}.with_item(PLAYERS[0], DRAW).with_item(PLAYERS[1], DRAW)
        elif BEATS[first] == second:
            {PAYOFF} = {PAYOFF}.with_item(PLAYERS[0], WIN).with_item(PLAYERS[1], LOSS)
        else:
            {PAYOFF} = {PAYOFF}.with_item(PLAYERS[0], LOSS).with_item(PLAYERS[1], WIN)
        for thrower in PLAYERS:
            {TURN} = {TURN}.with_item(thrower, False)
        """
    )
    declarer.definitions(create_rock_paper_scissors_definitions(), effects=True)
    declarer.leads_to(THROW, PythonRule(f"{HAND} = {HAND}.with_item({PLAYER}, {SHAPE})"))
    declarer.together(PythonRule(resolution))


def create_rock_paper_scissors_players() -> Players:
    """A and B; the turn map flags the players to act; the payoff map holds each one's payoff."""
    return Players(PLAYERS, TURN, PAYOFF)


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
