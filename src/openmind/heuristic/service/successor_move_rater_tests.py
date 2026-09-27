from collections.abc import Callable

from openmind.heuristic.model.node import Node
from openmind.heuristic.service.successor_move_rater import SuccessorMoveRater
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


class _Flat:
    """A position value saying every unfinished position is worth the same to everybody, so that what a
    finished one paid is the only thing that can tell two actions apart."""

    def __init__(self, worth: float = 0.0, players: int = 2) -> None:
        self._values = (worth,) * players

    def values(self, model: object, node: Node) -> tuple[float, ...] | None:
        return self._values


class _Silent:
    """A position value that knows nothing, which is not the same as saying nothing is worth anything."""

    def values(self, model: object, node: Node) -> tuple[float, ...] | None:
        return None


def test_the_winning_action_is_worth_what_the_game_paid_for_it(game: Game) -> None:
    """A finished position is worth what the game paid there and never a model's opinion of it. Every
    unfinished position here is worth the same, so what the game paid is the only thing separating them."""
    playing = game("tictactoe")
    almost = State.of(
        cell=Grid((3, 3), ("X", "X", None, "O", "O", None, None, None, None)),
        turn="X",
        payoff=Map.of({"X": None, "O": None}),
    )
    node = playing.node(almost)
    actions = playing.actions(almost, player="X")

    rated = SuccessorMoveRater(_Flat()).rate(None, node, actions, "X")

    held = {action: value for action, value in zip(actions, rated, strict=True)}
    winning = next(one for one in actions if one.parameters == (("col", 3), ("row", 1)))
    assert held[winning] == 1.0, "what the game paid the mover there"
    assert all(value == 0.0 for action, value in held.items() if action is not winning)


def test_it_knows_nothing_where_the_position_value_knows_nothing(game: Game) -> None:
    """None the whole way through: a rater whose valuer says nothing has nothing of its own to add."""
    playing = game("tictactoe")
    node = playing.node(playing.start())
    actions = playing.actions(node.state, player="X")

    rated = SuccessorMoveRater(_Silent()).rate(None, node, actions, "X")

    assert rated == (None,) * len(actions)


def test_it_rates_nothing_where_the_players_choose_at_once(game: Game) -> None:
    """What an action leads to is not settled by that action alone when somebody else is choosing at the same
    moment. Declining is the answer; pretending the others pass is not."""
    playing = game("rockpaperscissors")
    node = playing.node(playing.start())
    actions = playing.actions(node.state, player="A")

    rated = SuccessorMoveRater(_Flat()).rate(None, node, actions, "A")

    assert rated == (None,) * len(actions)


def test_it_rates_nothing_for_a_player_the_game_does_not_have(game: Game) -> None:
    playing = game("tictactoe")
    node = playing.node(playing.start())
    actions = playing.actions(node.state, player="X")

    assert SuccessorMoveRater(_Flat()).rate(None, node, actions, "nobody") == (None,) * len(actions)


def test_built_with_no_valuer_it_says_so_rather_than_rating(game: Game) -> None:
    """It fills the move task with a position value it was given; given none it knows nothing."""
    playing = game("tictactoe")
    node = playing.node(playing.start())
    actions = playing.actions(node.state, player="X")

    assert SuccessorMoveRater().rate(None, node, actions, "X") == (None,) * len(actions)


def test_a_caller_carrying_its_own_valuer_asks_through_it(game: Game) -> None:
    """A planner is handed its position value with its guidance, and it is not this rater's to assume."""
    playing = game("tictactoe")
    node = playing.node(playing.start())
    actions = playing.actions(node.state, player="X")

    rated = SuccessorMoveRater().through(_Flat(worth=0.5), None, node, actions, "X")

    assert rated == (0.5,) * len(actions), "the valuer given here, not the one it was built with"
