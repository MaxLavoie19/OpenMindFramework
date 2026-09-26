from collections.abc import Callable

from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.mapper.position_row_mapper import PositionRowMapper
from openmind.training.model.played_game import PlayedGame
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction

type Game = Callable[[str], RuleBasedGame]


def a_game(played: RuleBasedGame, payoffs=(1.0, 0.0), steps: int = 3) -> PlayedGame:
    """A game of that many moves, paying what it says."""
    state = played.start()
    states, actions = [state], []
    for column in range(1, steps + 1):
        action = Action("place", (("col", column), ("row", 1)))
        player = played.players().names[len(actions) % 2]
        actions.append(JointAction(((player, action),)))
        state = played.outcomes(state, action).outcomes[0][0]
        states.append(state)
    return PlayedGame(tuple(states), tuple(actions), payoffs)


def test_every_position_a_game_went_through_becomes_a_row_for_each_player(game: Game) -> None:
    """A position is worth something to each side, and a fit that only ever saw one side's view would learn a
    heuristic that could not be asked about the other."""
    played = game("tictactoe")

    rows = PositionRowMapper().to_rows(played, [a_game(played)])

    assert len(rows) == 4 * 2  # four positions, two players
    assert {row.player for row in rows} == {"X", "O"}


def test_a_position_is_valued_at_what_that_game_paid_that_player(game: Game) -> None:
    """A game's result is a rough thing to value a position by — a position can be winning and still lost by
    whoever reached it — but it is the one value a game states for certain."""
    played = game("tictactoe")

    rows = PositionRowMapper().to_rows(played, [a_game(played, payoffs=(1.0, 0.0))])

    assert {row.target for row in rows if row.player == "X"} == {1.0}
    assert {row.target for row in rows if row.player == "O"} == {0.0}


def test_a_game_that_paid_nobody_gives_no_rows(game: Game) -> None:
    """A game cut short before it ended has nothing to say about whether its positions were worth reaching,
    and valuing them at nothing would teach that every position it passed through was a loss."""
    played = game("tictactoe")

    assert PositionRowMapper().to_rows(played, [a_game(played, payoffs=())]) == ()


def test_a_game_paying_the_wrong_number_of_players_gives_no_rows(game: Game) -> None:
    """Zipping a payoff per player against a different count would either drop a player's rows silently or
    raise in the middle of a distillation."""
    played = game("tictactoe")

    assert PositionRowMapper().to_rows(played, [a_game(played, payoffs=(1.0,))]) == ()


def test_the_rows_of_several_games_are_pooled(game: Game) -> None:
    """A heuristic is fitted over many games, since one game says almost nothing about which positions are
    worth reaching."""
    played = game("tictactoe")

    rows = PositionRowMapper().to_rows(played, [a_game(played), a_game(played, payoffs=(0.0, 1.0))])

    assert len(rows) == 16
    assert {row.target for row in rows} == {0.0, 1.0}


def test_no_games_give_no_rows(game: Game) -> None:
    assert PositionRowMapper().to_rows(game("tictactoe"), []) == ()
