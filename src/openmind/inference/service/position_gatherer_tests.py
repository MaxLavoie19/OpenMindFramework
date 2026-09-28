import itertools
from collections.abc import Callable

from openmind.inference.service.position_gatherer import PositionGatherer
from openmind.world.model.joint_action import JointAction
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


def test_positions_are_reached_by_playing_the_game_from_where_it_starts(game: Game) -> None:
    played = game("tictactoe")

    positions = PositionGatherer().gather(played, 30, seed=1)

    assert len(positions) == 30
    assert positions[0] == played.start()
    assert len({position for position in positions}) == 30  # each one different


def test_a_walk_that_reaches_the_end_starts_again(game: Game) -> None:
    """Tic-tac-toe is over in nine moves, so thirty positions take several games."""
    played = game("tictactoe")

    positions = PositionGatherer().gather(played, 30, seed=1)

    assert any(not played.joint_actions(position) for position in positions)  # some are positions it is over in


def test_players_acting_at_once_are_moved_together(game: Game) -> None:
    """Rock paper scissors is over once both have thrown: moving one player at a time would ask the game what half a
    throw leads to."""
    played = game("rockpaperscissors")

    positions = PositionGatherer().gather(played, 8, seed=1)

    assert len(positions) > 1
    assert all(position == played.start() or position.model("payoff").values() != (None, None) for position in positions)


def test_a_game_nobody_can_act_in_gives_its_starting_position_alone(declared: Callable[..., RuleBasedGame]) -> None:
    from openmind.world.model.state import State

    settled = declared(State.of(turn="X"))

    assert PositionGatherer().gather(settled, 10) == (settled.start(),)


def test_a_walk_gives_neighbours_where_gathering_does_not(game: Game) -> None:
    """Gathering skips a position it has seen and starts again when a game ends, so two of its positions side
    by side need not be a move apart. Anything asking what one move changes needs the steps themselves."""
    played = game("tictactoe")

    walks = PositionGatherer().walk(played, 40, seed=1, walks=2)

    assert len(walks) == 2
    for walk in walks:
        assert len(walk) > 1, "a walk of one position is no walk"
        for at in range(1, len(walk)):
            assert walk[at] in _reachable(played, walk[at - 1]), "each position is one move from the one before"


def _reachable(played: RuleBasedGame, state: State) -> set[State]:
    """Every position one joint action leads to from there, so a claim of adjacency can be checked rather
    than assumed."""
    legal = played.joint_actions(state)
    players = played.players().names
    found: set[State] = set()
    for chosen in itertools.product(*[actions for _, actions in legal]):
        joint = JointAction(tuple((players[index], chosen[at]) for at, (index, _) in enumerate(legal)))
        found.update(outcome for outcome, _ in played.joint_outcomes(state, joint).outcomes)
    return found


def test_a_walk_stops_where_the_game_does(game: Game) -> None:
    """A walk that ends before its steps run out comes back short rather than being stitched to a fresh game:
    the step from the end of one game to the start of the next is not a step anything took."""
    played = game("tictactoe")

    walk = PositionGatherer().walk(played, 500, seed=2)[0]

    assert len(walk) < 500, "tic-tac-toe cannot go five hundred moves"
    assert not played.joint_actions(walk[-1]), "it stopped because the game did"


def test_asking_for_no_walks_walks_nothing(game: Game) -> None:
    assert PositionGatherer().walk(game("tictactoe"), 40, walks=0) == ()
