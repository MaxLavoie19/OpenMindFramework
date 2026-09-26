from collections.abc import Callable

import pytest

from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.service.game_replayer import GameReplayer
from openmind.world.model.action import Action

type Game = Callable[[str], RuleBasedGame]

PLAYED = (
    Action("place", (("col", 1), ("row", 1))),
    Action("place", (("col", 2), ("row", 1))),
    Action("place", (("col", 1), ("row", 2))),
)


def a_summary(**held) -> GameSummary:
    model = ModelDescription("a heuristic", "{}")
    fields = {
        "domain": "tictactoe",
        "kind": "self-play",
        "round": None,
        "number": 1,
        "seeds": (7, 11),
        "players": ("X", "O"),
        "models": (model, model),
        "payoffs": (1.0, 0.0),
        "plies": len(PLAYED),
        "actions": PLAYED,
    }
    return GameSummary(**{**fields, **held})


def test_a_game_comes_back_with_one_more_position_than_it_had_actions(game: Game) -> None:
    """A game is remembered by what was played in it, not by every position it went through — a chess game is
    a hundred positions of 64 squares and a hundred short strings."""
    played = game("tictactoe")

    positions = GameReplayer().positions(played, a_summary())

    assert len(positions) == len(PLAYED) + 1
    assert positions[0] == played.start()


def test_the_positions_come_back_as_they_were(game: Game) -> None:
    played = game("tictactoe")

    positions = GameReplayer().positions(played, a_summary())

    assert positions[1].model("cell").at((1, 1)) == "X"
    assert positions[2].model("cell").at((1, 2)) == "O"


def test_playing_it_again_draws_the_same_chances_it_drew_the_first_time(game: Game) -> None:
    """A game with chance in it is only replayable because its outcome seed was kept apart from its agent seed;
    drawn afresh, a page would show a game nobody played."""
    played = game("tictactoe")
    replayer = GameReplayer()

    assert replayer.positions(played, a_summary()) == replayer.positions(played, a_summary())


def test_a_game_remembered_without_an_outcome_seed_says_so(game: Game) -> None:
    """Silently replaying it with a fresh seed would show positions that never happened, and nothing on the
    page would say they hadn't."""
    played = game("tictactoe")

    with pytest.raises(ValueError, match="outcome seed"):
        GameReplayer().positions(played, a_summary(seeds=(7,)))


def test_a_game_where_several_acted_at_once_is_not_followed_past_that(
    game: Game, knowledge
) -> None:
    """The summary says what was played and not who played it, so where more than one player could act there
    is no way to tell whose action this was. The page then shows what the game paid without its positions,
    rather than guessing at an order and drawing a game nobody played."""
    from openmind.rbs.factory.rbs_factory import create_rule_based_game
    from openmind.rbs.service.game_relaxer import GameRelaxer

    game("tictactoe")
    # The game without its turn rule: both players may act in every position.
    at_once = create_rule_based_game(
        knowledge, GameRelaxer(knowledge).relax("tictactoe", "tictactoe without place is legal, 1")
    )

    positions = GameReplayer().positions(at_once, a_summary())

    assert positions == (at_once.start(),)


def test_it_follows_the_actions_it_was_given_without_checking_them_again(game: Game) -> None:
    """They came from a game that played them, so they were legal when they were played; solving for the legal
    actions at every ply to confirm it would cost a whole search to learn nothing."""
    played = game("tictactoe")
    twice = a_summary(actions=(PLAYED[0], PLAYED[0]))

    positions = GameReplayer().positions(played, twice)

    assert len(positions) == 3


def test_a_game_with_no_actions_is_just_where_it_started(game: Game) -> None:
    played = game("tictactoe")

    assert GameReplayer().positions(played, a_summary(actions=())) == (played.start(),)
