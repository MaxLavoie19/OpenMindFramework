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


def test_without_an_economy_every_term_the_fit_kept_is_written(game: Game, knowledge: KnowledgeBase) -> None:
    """The gate is something a caller turns on. Off, this writes what it wrote before there was a budget, so
    nothing that was working starts depending on an economy nobody asked for."""
    played = game("tictactoe")
    training, held_out = rows(played)

    generated = create_heuristic_finder().generate(
        played, training, held_out, settings(), HeuristicTarget(knowledge, "tictactoe")
    )

    assert len(generated.rules) - 1 == generated.chosen.terms_kept, "the constant, and every term the fit kept"


def test_on_a_thousandth_of_a_bit_a_round_only_the_constant_is_written(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """The second gate, after the price sweep. L1 decides what a term weighs; the budget decides whether it is
    admitted — so a fit that kept terms can still write none of them.

    The cheapest rule there is costs 1.5 bits to say, so at a thousandth of a bit a round no signal can ever
    afford anything, whatever the search happened to find. That is what makes this an exact claim rather than
    one that depends on a time-budgeted search turning something up."""
    played = game("tictactoe")
    training, held_out = rows(played)

    poor = create_heuristic_finder(allowance=0.001).generate(
        played, training, held_out, settings(), HeuristicTarget(knowledge, "tictactoe")
    )

    assert len(poor.rules) == 1, "the constant, and nothing anybody could pay for"


def test_a_budget_nothing_could_exhaust_writes_what_the_fit_kept(game: Game, knowledge: KnowledgeBase) -> None:
    """The other end of the same knob, and the proof the gate is wired rather than merely present: given more
    than every candidate costs put together, the economy writes exactly what there was no economy before.

    **How much comes through in between is pinned in `rule_admission_tests.py`, not here.** The search runs on
    a one-second budget and finds thirty-nine terms on an idle machine and none on a loaded one, so a test
    that asserted an ordering over allowances would be measuring the machine. Measured idle, for the record:
    twelve bits a round wrote two terms, thirty wrote seven, a hundred wrote sixteen and a thousand wrote all
    thirty-nine."""
    played = game("tictactoe")
    training, held_out = rows(played)

    rich = create_heuristic_finder().generate(
        played, training, held_out, settings(), HeuristicTarget(knowledge, "tictactoe")
    )
    spent = create_knowledge_base("unlimited", Path(mkdtemp()))
    lavish = create_heuristic_finder(allowance=1_000_000.0).generate(
        played, training, held_out, settings(), HeuristicTarget(spent, "tictactoe")
    )

    assert lavish.chosen.terms_kept == rich.chosen.terms_kept, "the same fit either way"
    assert len(lavish.rules) == lavish.chosen.terms_kept + 1, "and every term of it was bought"
