from collections.abc import Callable

from openmind.inference.service.position_gatherer import PositionGatherer
from openmind.rbs.service.rule_based_game import RuleBasedGame

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
