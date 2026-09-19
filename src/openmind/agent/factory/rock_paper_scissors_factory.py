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
    UNSET,
    WIN,
)
from openmind.game.service.game_declarer import GameDeclarer
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.map import Map
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.constant.players_constant import PLAYER
from openmind.world.model.players import Players
from openmind.world.model.state import State


def create_rock_paper_scissors_initial_state() -> State:
    """The hand map holding no hand thrown; the payoff map holding no payoff."""
    return (
        StateBuilder()
        .with_model(HAND, Map.of(dict.fromkeys(PLAYERS, UNSET)))
        .with_model(PAYOFF, Map.of(dict.fromkeys(PLAYERS, UNSET)))
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


def declare_rock_paper_scissors_moves(declarer: GameDeclarer) -> None:
    """A player who hasn't thrown throws a shape, both at once."""
    declarer.values(THROW, SHAPE, PythonRule(repr(SHAPES)))
    declarer.constraints(THROW, PythonRule(f"{HAND}[{PLAYER}] is {UNSET!r}"))


def declare_rock_paper_scissors_effects(declarer: GameDeclarer) -> None:
    """Each throw sets its player's hand; then the resolution compares the hands and sets the payoffs, after which no
    player has an action left."""
    resolution = textwrap.dedent(
        f"""\
        first, second = {HAND}[PLAYERS[0]], {HAND}[PLAYERS[1]]
        if first == second:
            {PAYOFF} = {PAYOFF}.with_item(PLAYERS[0], DRAW).with_item(PLAYERS[1], DRAW)
        elif BEATS[first] == second:
            {PAYOFF} = {PAYOFF}.with_item(PLAYERS[0], WIN).with_item(PLAYERS[1], LOSS)
        else:
            {PAYOFF} = {PAYOFF}.with_item(PLAYERS[0], LOSS).with_item(PLAYERS[1], WIN)
        """
    )
    declarer.definitions(create_rock_paper_scissors_definitions(), effects=True)
    declarer.leads_to(THROW, PythonRule(f"{HAND} = {HAND}.with_item({PLAYER}, {SHAPE})"))
    declarer.together(PythonRule(resolution))


def create_rock_paper_scissors_players() -> Players:
    """A and B; the payoff map holds each one's payoff."""
    return Players(PLAYERS, PAYOFF)


def declare_rock_paper_scissors(knowledge_base: KnowledgeBase) -> str:
    """Declares rock paper scissors' rules and gives back the context they were declared under: A and B throw at once;
    rock beats scissors, paper beats rock, scissors beat paper."""
    declarer = GameDeclarer(knowledge_base, NAME)
    declarer.starts_at(create_rock_paper_scissors_initial_state())
    declarer.played_by(create_rock_paper_scissors_players())
    declarer.definitions(create_rock_paper_scissors_definitions())
    declare_rock_paper_scissors_moves(declarer)
    declare_rock_paper_scissors_effects(declarer)
    return declarer.done()


def declare_rock_paper_scissors_named(name: str, knowledge_base: KnowledgeBase) -> str:
    """Declares rock paper scissors from its registered name, which has no variant; any other raises ValueError."""
    if name != NAME:
        raise ValueError(f"{NAME} has no variant: {name!r}")
    return declare_rock_paper_scissors(knowledge_base)
