from collections.abc import Callable
from pathlib import Path
from tempfile import mkdtemp

from openmind.inference.service.accuracy_scorer import AccuracyScorer
from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.service.model_registry import ModelRegistry
from openmind.rbs.factory.rbs_factory import create_heuristic_finder
from openmind.rbs.model.heuristic_target import HeuristicTarget
from openmind.rbs.model.position_row import PositionRow
from openmind.rbs.model.value_settings import ValueSettings
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rbs.service.heuristic_finder import PRICED_RULESET

type Game = Callable[[str], RuleBasedGame]

PRICES = (0.0, 0.05)


def settings(**held) -> ValueSettings:
    return ValueSettings(
        prices=PRICES, max_steps=60, tolerance=1e-4, seconds=1.0, memory_bytes=200_000_000, **held
    )


def rows(played: RuleBasedGame) -> tuple[list[PositionRow], list[PositionRow]]:
    """A handful of positions valued differently, so there is something to fit and something to choose on."""
    start = played.start()
    reached = [start]
    for action in played.actions(start)[:5]:
        reached.append(played.outcomes(start, action).outcomes[0][0])
    valued = [PositionRow(state, "X", float(number % 3)) for number, state in enumerate(reached)]
    return valued[:4], valued[4:]


def test_only_the_chosen_price_is_declared_unless_every_price_is_asked_for(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """What a caller asked for before is what it still gets: one model of the task, the fit the held-out rows
    preferred, under the task's own name."""
    played = game("tictactoe")
    training, held_out = rows(played)

    generated = create_heuristic_finder().generate(
        played, training, held_out, settings(), HeuristicTarget(knowledge, "tictactoe")
    )

    context = knowledge.context_named("tictactoe")
    assert generated.others == ()
    assert [model.name for model in ModelRegistry(AccuracyScorer()).of_task(knowledge, context.id, POSITION_VALUE)] == [
        POSITION_VALUE
    ]


def test_every_price_kept_is_a_model_of_its_own_to_be_played(game: Game, knowledge: KnowledgeBase) -> None:
    """The sweep fits at every price and throws all but one away, and a sparse fit and a dense one are different
    heuristics rather than one heuristic at two settings.

    Which of them is worth playing with is a question no loss answers, so where something means to find out by
    playing, each is kept as a model of the same task — which is what lets a bandit rank them."""
    played = game("tictactoe")
    training, held_out = rows(played)

    generated = create_heuristic_finder().generate(
        played, training, held_out, settings(keep_every_price=True), HeuristicTarget(knowledge, "tictactoe")
    )

    context = knowledge.context_named("tictactoe")
    found = {model.name for model in ModelRegistry(AccuracyScorer()).of_task(knowledge, context.id, POSITION_VALUE)}
    assert len(found) == len(PRICES)
    assert POSITION_VALUE in found
    assert len(generated.others) == len(PRICES) - 1
    assert all(named in found for named, _ in generated.others)
    assert all(
        named == PRICED_RULESET.format(task=POSITION_VALUE, price=f"{fit.price:g}")
        for (named, _), fit in zip(generated.others, [one for one in generated.fits if one is not generated.chosen], strict=True)
    )


def test_rules_fitted_in_another_store_can_be_taken_into_this_one(game: Game, knowledge: KnowledgeBase) -> None:
    """A worker that ponders the game it has just played ponders in a store of its own, because a second
    writer to the run's store is a corrupted store. What it settles crosses back as rules and weights, and
    arrives here as a ruleset of the task, registered as a model so it can be drawn to play."""
    played = game("tictactoe")
    training, held_out = rows(played)
    elsewhere = create_knowledge_base("elsewhere", Path(mkdtemp()))
    create_heuristic_finder().generate(
        played, training, held_out, settings(), HeuristicTarget(elsewhere, "tictactoe")
    )
    fitted = elsewhere.ruleset_rules(elsewhere.ruleset_named(elsewhere.context_named("tictactoe").id, POSITION_VALUE).id)

    taken = create_heuristic_finder().adopt(HeuristicTarget(knowledge, "tictactoe", name="what a worker settled"), fitted)

    context = knowledge.context_named("tictactoe")
    assert [one.name for one in taken] == [record.name for record, _ in fitted]
    assert "what a worker settled" in {
        model.name for model in ModelRegistry(AccuracyScorer()).of_task(knowledge, context.id, POSITION_VALUE)
    }
    landed = knowledge.ruleset_rules(knowledge.ruleset_named(context.id, "what a worker settled").id)
    assert [weight for _, weight in landed] == [weight for _, weight in fitted]


def test_taking_the_same_rules_in_twice_revises_them_rather_than_doubling_them(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """A worker fitting the same term every game it plays would otherwise grow a copy of it every game."""
    played = game("tictactoe")
    training, held_out = rows(played)
    elsewhere = create_knowledge_base("elsewhere", Path(mkdtemp()))
    create_heuristic_finder().generate(
        played, training, held_out, settings(), HeuristicTarget(elsewhere, "tictactoe")
    )
    fitted = elsewhere.ruleset_rules(elsewhere.ruleset_named(elsewhere.context_named("tictactoe").id, POSITION_VALUE).id)
    target = HeuristicTarget(knowledge, "tictactoe", name="what a worker settled")

    create_heuristic_finder().adopt(target, fitted)
    create_heuristic_finder().adopt(target, fitted)

    context = knowledge.context_named("tictactoe")
    landed = knowledge.ruleset_rules(knowledge.ruleset_named(context.id, "what a worker settled").id)
    assert len(landed) == len(fitted)
