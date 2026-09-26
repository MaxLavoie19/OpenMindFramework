import itertools
import logging
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

#: What a player may spend on one step here.
#:
#: **A hundredth of the previous budget, because the budget now means something.** A planner turns its seconds
#: into nodes by dividing by what a reading has been costing, and until readings were measured every model was
#: taken to cost a millisecond — so a tenth of a second bought eighty nodes. Measured, a tic-tac-toe heuristic
#: costs about three microseconds, the same tenth of a second buys twenty-six thousand nodes, and these tests
#: went from four seconds to nearly three minutes. Nothing was wrong: the search was finally spending what it
#: was given. What these tests want is a few games played, not a budget honoured, so they ask for less.
PLAY = SelfPlaySettings(seconds=0.003)


def arm(knowledge: KnowledgeBase, heuristic: Linked, name: str, rule: str) -> Arm:
    """A heuristic of its own, in a ruleset of its own, registered as a model of the position value task.

    **A model of the position value task, and not of a task named after itself.** It used to register the arm's
    own name as the task, so `Outfitter` — which loads models of the tasks it knows — found nothing and every
    game here was played with no heuristic at all. The bandit's bookkeeping was exercised and the heuristics it
    is for never were, which is exactly the shape of test that passes while the thing under it does nothing."""
    heuristic("tictactoe", name, PythonRule(rule), 1.0, task=POSITION_VALUE, ruleset_name=name)
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


def test_with_no_number_of_games_it_plays_until_something_stops_it(
    game: Game, knowledge: KnowledgeBase, heuristic: Linked
) -> None:
    """A bandit has no point at which it is finished — it is a way of spending attention, not a procedure that
    terminates. Stopped from outside, what it learned is returned rather than lost."""
    played = game("tictactoe")
    arms = (
        arm(knowledge, heuristic, "the first one", "1.0"),
        arm(knowledge, heuristic, "the second one", "0.0"),
    )
    ranker = create_heuristic_ranker()
    playing = ranker._self_play.play
    stop_after = 4
    counted = itertools.count()

    def stopped(*arguments, **named):
        if next(counted) >= stop_after:
            raise KeyboardInterrupt
        return playing(*arguments, **named)

    ranker._self_play.play = stopped  # type: ignore[method-assign]

    ranked = ranker.rank(knowledge, played, arms, RankingSettings(PLAY, games=None, seed=1))

    assert sum(score.games for score in ranked) == stop_after * 2
    assert all(score.games > 0 for score in ranked)


def test_the_standings_are_written_out_along_the_way(
    game: Game, knowledge: KnowledgeBase, heuristic: Linked, caplog
) -> None:
    """A run with no end has to be readable while it runs, and a table of results has a header row."""
    caplog.set_level(logging.INFO, logger="openmind.training.service.heuristic_ranker")
    played = game("tictactoe")
    arms = (
        arm(knowledge, heuristic, "the first one", "1.0"),
        arm(knowledge, heuristic, "the second one", "0.0"),
    )

    create_heuristic_ranker().rank(knowledge, played, arms, RankingSettings(PLAY, games=4, seed=1, standings=2))

    headers = [one for one in caplog.messages if "heuristic" in one and "per game" in one and "bound" in one]
    assert len(headers) == 2
    assert any("the first one" in one for one in caplog.messages)
