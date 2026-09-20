from collections.abc import Callable

import pytest

from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.model.rule_record import RuleRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.factory.model_factory import create_model_registry
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule
from openmind.training.factory.training_factory import create_heuristic_ranker
from openmind.training.model.arm import Arm
from openmind.training.model.ranking_settings import RankingSettings
from openmind.training.model.self_play_settings import SelfPlaySettings

type Game = Callable[[str], RuleBasedGame]
type Linked = Callable[..., RuleRecord]

PLAY = SelfPlaySettings(seconds=0.1)


def arm(knowledge: KnowledgeBase, heuristic: Linked, name: str, rule: str) -> Arm:
    """A heuristic of its own, in a ruleset of its own, registered as a model of the position value task."""
    heuristic("tictactoe", name, PythonRule(rule), 1.0, task=name)
    context = knowledge.context_named("tictactoe")
    ruleset = knowledge.ruleset_named(context.id, name)  # type: ignore[union-attr]
    return Arm(name, create_model_registry().register_ruleset(knowledge, ruleset, name))  # type: ignore[arg-type]


def test_heuristics_are_ranked_by_what_playing_with_them_paid(
    game: Game, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    """Every game is between two heuristics and counts for both, and what ranks one above another is the points its
    games paid it.

    Which of two heuristics is better isn't what this pins: telling two close ones apart takes far more games than a
    test can play, and six games of tic-tac-toe say nothing about the centre."""
    played = game("tictactoe")
    centre = arm(knowledge, heuristic, "taking the centre", "1.0 if cell[2, 2] == me else 0.0")
    edge = arm(knowledge, heuristic, "avoiding the centre", "0.0 if cell[2, 2] == me else 1.0")

    ranked = create_heuristic_ranker().rank(knowledge, played, (centre, edge), RankingSettings(PLAY, games=6, seed=1))

    assert sum(score.games for score in ranked) == 12  # every game counts for both sides
    assert sum(score.points for score in ranked) == 6.0  # and pays a point between them
    assert ranked[0].points >= ranked[1].points and ranked[0].mean >= ranked[1].mean


def test_every_heuristic_gets_played_before_any_is_played_twice(
    game: Game, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    """UCB1 tries what it has never tried first, so a heuristic isn't written off before it has played."""
    played = game("tictactoe")
    arms = tuple(
        arm(knowledge, heuristic, f"the {name} one", rule)
        for name, rule in (("first", "1.0"), ("second", "0.0"), ("third", "0.5"))
    )

    ranked = create_heuristic_ranker().rank(knowledge, played, arms, RankingSettings(PLAY, games=3, seed=1))

    assert all(score.games > 0 for score in ranked)


def test_ranking_one_heuristic_alone_says_so(game: Game, knowledge: KnowledgeBase, heuristic: Linked) -> None:
    played = game("tictactoe")
    alone = arm(knowledge, heuristic, "the only one", "1.0")

    with pytest.raises(ValueError, match="at least two"):
        create_heuristic_ranker().rank(knowledge, played, (alone,), RankingSettings(PLAY, games=2))
