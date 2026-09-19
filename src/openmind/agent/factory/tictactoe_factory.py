import textwrap

from openmind.agent.constant.agent_constant import FLAGGED
from openmind.agent.constant.tictactoe_constant import (
    CELL,
    COL,
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
from openmind.agent.model.tictactoe_variant import TicTacToeVariant
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.players import Players
from openmind.world.model.state import State


def create_tictactoe_initial_state(variant: TicTacToeVariant = STANDARD) -> State:
    """The cell grid, the variant's height by its width, every cell empty, row 1 at the top; X to play; the payoff map
    holding no payoff for either player."""
    _check(variant)
    return (
        StateBuilder()
        .with_model(CELL, Grid.filled((variant.height, variant.width), EMPTY))
        .with_model(TURN, PLAYERS[0])
        .with_model(PAYOFF, Map.of(dict.fromkeys(PLAYERS, UNSET)))
        .build()
    )


def create_tictactoe_definitions(variant: TicTacToeVariant = STANDARD) -> PythonRule:
    """The names every tic-tac-toe rule sees: the variant's WIDTH, HEIGHT and LINE, the PLAYERS, the WIN, DRAW and LOSS
    payoffs, other(player), and LINES_THROUGH[row, col], every line of LINE cells through a cell that fits the grid,
    worked out once from the lines of an empty grid of the variant's size."""
    _check(variant)
    return PythonRule(
        textwrap.dedent(
            f"""\
            WIDTH, HEIGHT, LINE = {variant.width}, {variant.height}, {variant.line}
            PLAYERS = {PLAYERS!r}
            WIN, DRAW, LOSS = {WIN!r}, {DRAW!r}, {LOSS!r}


            def other(player):
                return PLAYERS[1] if player == PLAYERS[0] else PLAYERS[0]


            def lines_through_every_cell():
                board = Grid.filled((HEIGHT, WIDTH), None)
                lines = {{where: [] for where in board}}
                for line in board.lines(LINE):
                    for where in line:
                        lines[where].append(line)
                return lines


            LINES_THROUGH = lines_through_every_cell()
            """
        )
    )


def declare_tictactoe_moves(declarer: RuleDeclarer, variant: TicTacToeVariant = STANDARD) -> None:
    """While no payoff is set: place a mark on an empty cell or, with gravity, drop it in a column whose top cell is
    empty."""
    _check(variant)
    no_payoff_set = tuple(PythonRule(f"{PAYOFF}[{player!r}] is {UNSET!r}") for player in PLAYERS)
    columns = PythonRule(repr(tuple(range(1, variant.width + 1))))
    if variant.gravity:
        declarer.values(DROP, COL, columns)
        declarer.constraints(DROP, *no_payoff_set, PythonRule(f"{CELL}[1, {COL}] is {EMPTY!r}"))
        return
    declarer.values(PLACE, ROW, PythonRule(repr(tuple(range(1, variant.height + 1)))))
    declarer.values(PLACE, COL, columns)
    declarer.constraints(PLACE, *no_payoff_set, PythonRule(f"{CELL}[{ROW}, {COL}] is {EMPTY!r}"))


def declare_tictactoe_effects(declarer: RuleDeclarer, variant: TicTacToeVariant = STANDARD) -> None:
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
        full = f"all(mark is not {EMPTY!r} for mark in {CELL}.cells)"
    effects = landing + textwrap.dedent(
        f"""\
        {CELL} = {CELL}.placed(({ROW}, {COL}), {TURN})
        if any(all({CELL}[where] == {TURN} for where in line) for line in LINES_THROUGH[{ROW}, {COL}]):
            {PAYOFF} = {PAYOFF}.with_item({TURN}, WIN).with_item(other({TURN}), LOSS)
        elif {full}:
            {PAYOFF} = {PAYOFF}.with_item({first!r}, DRAW).with_item({second!r}, DRAW)
        {TURN} = other({TURN})
        """
    )
    declarer.definitions(create_tictactoe_definitions(variant), effects=True)
    declarer.leads_to(action, PythonRule(effects))


def create_tictactoe_timeout() -> PythonRule:
    """On a clock, the player whose time ran out loses and the other wins."""
    return PythonRule(f"{PAYOFF} = {PAYOFF}.with_item({FLAGGED}, LOSS).with_item(other({FLAGGED}), WIN)")


def create_tictactoe_players() -> Players:
    """X and O; turn names the player to act; the payoff map holds each one's payoff."""
    return Players(PLAYERS, TURN, PAYOFF)


def declare_tictactoe(
    knowledge_base: KnowledgeBase, variant: TicTacToeVariant = STANDARD, weight: float = 1.0
) -> str:
    """Declares tic-tac-toe's rules, or one of its variants', into the knowledge base, and gives back the context they
    were declared under: "tictactoe" for the standard game, "tictactoe/<variant>" for any other."""
    context = NAME if variant == STANDARD else SEPARATOR.join((NAME, variant.name))
    declarer = RuleDeclarer(knowledge_base, context, weight)
    declarer.starts_at(create_tictactoe_initial_state(variant))
    declarer.played_by(create_tictactoe_players())
    declarer.empty(CELL, EMPTY)
    declarer.definitions(create_tictactoe_definitions(variant))
    declare_tictactoe_moves(declarer, variant)
    declare_tictactoe_effects(declarer, variant)
    declarer.timeout(create_tictactoe_timeout())
    return declarer.done()


def _check(variant: TicTacToeVariant) -> None:
    if variant.width < 1 or variant.height < 1 or not 1 <= variant.line <= max(variant.width, variant.height):
        raise ValueError(
            "A tic-tac-toe variant needs a width and a height of at least 1 and a line from 1 to its longer side, "
            f"not {variant!r}"
        )
