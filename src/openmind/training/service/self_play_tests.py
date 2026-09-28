from collections.abc import Callable

from openmind.agent.service.game_memory import GameMemory
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.model.guidance import Guidance
from openmind.training.factory.training_factory import create_self_play
from openmind.training.service.self_play import CUT_SHORT, UNSTATED
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


def test_each_side_is_remembered_with_the_heuristic_it_actually_played(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """A game between two heuristics is only worth playing if it records which side played which.

    This used to remember one description of the *game* for both players, so two sides that judged differently
    were recorded as having judged alike and nothing afterwards could say which had won. The rules each side
    judged with come back with it, because a name says which won and the rules say why."""
    from openmind.agent.service.outfitter import Outfitter
    from openmind.knowledge.constant.task_constant import POSITION_VALUE
    from openmind.model.factory.model_factory import create_model_registry
    from openmind.rbs.factory.rbs_factory import create_rule_heuristic
    from openmind.rule.model.python_rule import PythonRule

    played = game("tictactoe")
    outfitter = Outfitter(create_rule_heuristic())
    filled = {}
    for named, source in (("the centre one", "cell[2, 2] == me"), ("the corner one", "cell[1, 1] == me")):
        heuristic("tictactoe", named, PythonRule(source), 0.75, task=POSITION_VALUE, ruleset_name=named)
        ruleset = knowledge.ruleset_named(knowledge.context_named("tictactoe").id, named)
        record = create_model_registry().register_ruleset(knowledge, ruleset, named)
        filled[named] = outfitter.filled(knowledge, record)

    create_self_play().play(
        knowledge,
        played,
        {"X": Guidance("X", filled["the centre one"]), "O": Guidance("O", filled["the corner one"])},
        SelfPlaySettings(games=1, seconds=0.2, seed=1),
    )

    summary = GameMemory(knowledge).games()[-1]
    assert [model.name for model in summary.models] == ["the centre one", "the corner one"]
    assert summary.models[0].id != summary.models[1].id
    assert "cell[2, 2] == me" in summary.models[0].text and "cell[1, 1] == me" in summary.models[1].text  # the rules, readable


def test_a_game_says_why_it_stopped_even_where_its_rules_do_not(game: Game, knowledge: KnowledgeBase) -> None:
    """Three ways to stop and they were one word between them, which was no word at all. A game cut short did
    not end, and reading it as one that did is how a run of abandoned games looks like a run of played-out
    ones."""
    played = game("tictactoe")

    whole = create_self_play().play_game(knowledge, played, Guidance("X"), SelfPlaySettings(seconds=0.003, seed=1))
    cut = create_self_play().play_game(
        knowledge, played, Guidance("X"), SelfPlaySettings(seconds=0.003, seed=1, steps=2)
    )

    assert cut.ending == CUT_SHORT, "the caller stopped it, and the game did not"
    assert cut.steps == 2
    assert whole.ending == UNSTATED, "it ended, and tic-tac-toe's rules say nothing about why"
    assert whole.steps > 2
