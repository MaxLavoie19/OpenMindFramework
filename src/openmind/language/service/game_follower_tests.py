from collections.abc import Callable, Sequence

from openmind.language.model.happening import Happening
from openmind.language.service.game_follower import NOWHERE, REFUSED, SEVERAL, GameFollower
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.model.action import Action
from openmind.statement.model.change import Placed
from openmind.world.model.state import State

type Game = Callable[[str], object]


class _Named:
    """A decoder that reads a notation as whichever happenings a test says it names.

    Stood in for rather than learned, because what is pinned here is the walking: what the follower does with
    one reading, with none, and with several. How a notation comes to name a happening is the decoder's own
    question and it has its own tests."""

    def __init__(self, naming: dict[str, Sequence[int]]) -> None:
        self._naming = naming

    def read(self, state, said, couplings, among, grammar, surely=0.9) -> tuple[Happening, ...]:
        return tuple(among[at] for at in self._naming.get(said, ()) if at < len(among))


def _placing(column: int) -> tuple[Action, Happening]:
    """Taking a cell of the top row, and what that does."""
    action = Action("place", (("col", column), ("row", 1)))
    return action, Happening((Placed("cell", (1, column), "X"),))


def _board(taken: tuple[int, ...] = ()) -> State:
    values = ["X" if at + 1 in taken else None for at in range(3)] + [None] * 6
    return State.of(cell=Grid((3, 3), tuple(values)), turn="X", payoff=Map.of({"X": None, "O": None}))


def _offers(state: State) -> tuple[tuple[Action, Happening], ...]:
    held = state.model("cell")
    return tuple(_placing(column) for column in (1, 2, 3) if held[1, column] is None)


def _after(state: State, action: Action) -> State:
    taken = tuple(column for column in (1, 2, 3) if state.model("cell")[1, column] is not None)
    return _board((*taken, dict(action.parameters)["col"]))


def test_a_game_somebody_else_wrote_down_is_followed_to_its_positions() -> None:
    """The whole point: a game nobody here played becomes positions to learn from."""
    follower = GameFollower(_Named({"a": (0,), "b": (0,), "c": (0,)}))

    following = follower.follow(_board(), ("a", "b", "c"), _offers, _after, (), None)

    assert following.whole
    assert following.followed == 3
    assert len(following.positions) == 4, "one more position than moves, counting where it started"


def test_a_notation_the_rules_refuse_stops_it_and_says_so() -> None:
    """The one mistake nothing else reports. A rule too tight never shows up in play, because the agent
    simply never makes the move it wrongly refuses — but somebody else played it."""
    follower = GameFollower(_Named({"a": (0,), "b": ()}))

    following = follower.follow(_board(), ("a", "b"), _offers, _after, (), None)

    assert following.stopped == REFUSED
    assert following.followed == 1, "it stops rather than guessing the rest"
    assert following.read == 1


def test_a_notation_naming_several_moves_stops_it_and_says_something_else() -> None:
    """A different fault wanting opposite work: the rules are fine and the reading cannot tell two moves
    apart. A count of positions would hide which it was."""
    follower = GameFollower(_Named({"a": (0, 1)}))

    following = follower.follow(_board(), ("a",), _offers, _after, (), None)

    assert following.stopped == SEVERAL
    assert following.followed == 0


def test_several_happenings_of_one_action_are_still_one_move() -> None:
    """A move drawn twice over is one move, so what is counted is the actions and not the drawings."""
    action, happening = _placing(1)
    follower = GameFollower(_Named({"a": (0, 1)}))

    following = follower.follow(
        _board(), ("a",), lambda state: ((action, happening), (action, happening)), _after, (), None
    )

    assert following.whole, "both readings are the same move"
    assert following.actions == (action,)


def test_a_position_where_nothing_can_happen_ends_the_following() -> None:
    """How a finished game ends, rather than a failure to report."""
    follower = GameFollower(_Named({"a": (0,)}))

    following = follower.follow(_board(), ("a",), lambda state: (), _after, (), None)

    assert following.stopped == NOWHERE
    assert following.positions == (_board(),)


def test_nothing_written_is_a_game_followed_whole() -> None:
    """Vacuously, and it must not read as a failure: there was nothing to disagree with."""
    following = GameFollower(_Named({})).follow(_board(), (), _offers, _after, (), None)

    assert following.whole
    assert following.followed == 0
