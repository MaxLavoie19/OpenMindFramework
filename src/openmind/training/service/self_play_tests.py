from collections.abc import Callable

from openmind.agent.service.game_memory import GameMemory
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.guidance import Guidance
from openmind.training.factory.training_factory import create_self_play
from openmind.training.model.self_play_settings import SelfPlaySettings
from openmind.training.service.game_replayer import GameReplayer

type Game = Callable[[str], RuleBasedGame]

SETTINGS = SelfPlaySettings(games=2, seconds=0.2, seed=1)


def test_a_game_is_played_to_its_end_and_says_what_it_paid(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")

    games = create_self_play().play(knowledge, played, Guidance("X"), SETTINGS)

    assert len(games) == 2
    for one in games:
        assert not played.joint_actions(one.states[-1])  # it ran until nobody could act
        assert len(one.payoffs) == 2 and len(one.states) == one.steps + 1
        assert one.decisive == (len(set(one.payoffs)) > 1)


def test_a_game_cut_short_by_its_steps_pays_nobody(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")

    (one,) = create_self_play().play(
        knowledge, played, Guidance("X"), SelfPlaySettings(games=1, seconds=0.2, steps=2, seed=1)
    )

    assert one.steps == 2 and one.payoffs == ()
    assert not one.decisive
    assert GameMemory(knowledge).games() == ()  # what is remembered is finished games


def test_every_game_is_remembered_with_what_was_played_and_what_it_paid(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")

    games = create_self_play().play(knowledge, played, Guidance("X"), SETTINGS)

    remembered = GameMemory(knowledge).games()
    assert len(remembered) == 2
    assert [summary.payoffs for summary in remembered] == [one.payoffs for one in games]
    assert all(summary.plies == one.steps for summary, one in zip(remembered, games, strict=True))
    assert all(summary.seeds == (one.agent_seed, one.outcome_seed) for summary, one in zip(remembered, games, strict=True))


def test_a_remembered_game_is_played_again_exactly_from_what_was_played(game: Game, knowledge: KnowledgeBase) -> None:
    """Its two streams of chance are kept apart, so its actions and its outcome seed bring its positions back."""
    played = game("tictactoe")
    (one,) = create_self_play().play(knowledge, played, Guidance("X"), SelfPlaySettings(games=1, seconds=0.2, seed=3))

    (summary,) = GameMemory(knowledge).games()
    positions = GameReplayer().positions(played, summary)

    assert positions == one.states
