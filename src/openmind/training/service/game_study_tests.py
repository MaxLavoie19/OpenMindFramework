from collections.abc import Callable

from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.guidance import Guidance
from openmind.training.factory.training_factory import create_game_study, create_self_play
from openmind.training.model.self_play_settings import SelfPlaySettings

type Game = Callable[[str], RuleBasedGame]

PLAY = SelfPlaySettings(games=2, seconds=0.003, seed=1)


def test_the_positions_of_games_played_come_back_to_be_learned_from(game: Game, knowledge: KnowledgeBase) -> None:
    """The point of the whole thing: a game is written down while it is played, because a move has to come
    back in time, and read back afterwards when nobody is waiting."""
    played = game("tictactoe")
    create_self_play().play(knowledge, played, Guidance("X"), PLAY)

    positions = create_game_study().positions(knowledge, played)

    assert positions, "games were played and remembered, so there is something to study"
    assert positions[0] == played.start(), "a game is studied from where it started"


def test_a_studied_position_can_be_put_to_the_game_exactly_as_a_walked_to_one_can(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """Why watching is not a harder problem than walking: the game is still there at study time, so what it
    allows at a remembered position is as askable as at a position a scout found."""
    played = game("tictactoe")
    create_self_play().play(knowledge, played, Guidance("X"), PLAY)

    positions = create_game_study().positions(knowledge, played)

    asked = [one for one in positions if played.joint_actions(one)]
    assert asked, "positions the game still has something to say about"
    assert all(played.actions(one, player=played.acting_player(one)) for one in asked)


def test_nothing_played_is_nothing_to_study(game: Game, knowledge: KnowledgeBase) -> None:
    assert create_game_study().positions(knowledge, game("tictactoe")) == ()


def test_only_the_last_games_are_studied_where_a_caller_asks_for_that(game: Game, knowledge: KnowledgeBase) -> None:
    """The last, because a game played recently was played with whatever is believed now."""
    played = game("tictactoe")
    create_self_play().play(knowledge, played, Guidance("X"), SelfPlaySettings(games=3, seconds=0.003, seed=1))
    study = create_game_study()

    every = study.positions(knowledge, played)
    recent = study.positions(knowledge, played, most=1)

    assert 0 < len(recent) < len(every)


def test_a_kind_of_game_can_be_studied_on_its_own(game: Game, knowledge: KnowledgeBase) -> None:
    """A knowledge base holds games of several kinds, and a study of one kind should not drag in the others."""
    played = game("tictactoe")
    create_self_play().play(knowledge, played, Guidance("X"), PLAY)
    study = create_game_study()

    assert study.positions(knowledge, played, kind="nothing of that kind") == ()
    assert study.positions(knowledge, played) != ()


def test_a_studied_position_carries_who_played_and_how_it_ended(game: Game, knowledge: KnowledgeBase) -> None:
    """Both were written down when the game ended and a bare position threw them away on the way out. A
    position from a game one heuristic won is evidence about that heuristic."""
    played = game("tictactoe")
    create_self_play().play(knowledge, played, Guidance("X"), PLAY)

    studied = create_game_study().studied(knowledge, played)

    assert studied
    assert all(one.players == played.players().names for one in studied), "who sat where"
    assert all(len(one.played_with) == len(one.players) for one in studied), "and what each played with"
    assert all(one.ending for one in studied), "every game says why it stopped, in one of three ways"
    assert {one.game for one in studied}, "each knows which game it came from"


def test_a_position_knows_how_far_into_its_game_it_is(game: Game, knowledge: KnowledgeBase) -> None:
    """So the openings of a thousand games can be told from the endings without replaying anything."""
    played = game("tictactoe")
    create_self_play().play(knowledge, played, Guidance("X"), SelfPlaySettings(games=1, seconds=0.003, seed=1))

    studied = create_game_study().studied(knowledge, played)

    assert [one.ply for one in studied] == list(range(len(studied))), "counted from the start at nought"
    assert studied[0].state == played.start()


def test_what_the_winner_played_with_is_read_off_the_position(game: Game, knowledge: KnowledgeBase) -> None:
    """The whole reason each side's own heuristic is remembered: a game between two of them says which won."""
    played = game("tictactoe")
    create_self_play().play(knowledge, played, Guidance("X"), PLAY)

    studied = create_game_study().studied(knowledge, played)

    for one in studied:
        if one.decisive:
            assert one.won_by() in one.played_with
        else:
            assert one.won_by() == "", "a drawn game names no winner rather than naming the first player"
