from collections.abc import Callable

import pytest

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.goal import Goal
from openmind.knowledge.model.preference import Preference
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.structure.model.map import Map
from openmind.utility.factory.utility_factory import create_even_binner, create_utility
from openmind.utility.model.bin import Bin
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


def paid(**payoffs: float | None) -> State:
    return State.of(payoff=Map.of(dict(payoffs)))


def outcomes(*paid_with_chance: tuple[State, float]) -> OutcomeDistribution:
    return OutcomeDistribution(tuple(paid_with_chance))


def test_a_move_is_worth_its_outcomes_payoffs_weighed_by_their_probabilities(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    won, drawn = paid(X=1.0, O=0.0), paid(X=0.5, O=0.5)

    worth = create_utility().of(knowledge, node, outcomes((won, 0.25), (drawn, 0.75)), "X", "payoff")

    assert worth == pytest.approx(0.25 * 1.0 + 0.75 * 0.5)


def test_an_outcome_nothing_can_be_read_from_is_left_out_and_a_move_of_those_alone_is_worth_nothing(
    game: Game, knowledge: KnowledgeBase
) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    unfinished = paid(X=None, O=None)

    assert create_utility().of(knowledge, node, outcomes((unfinished, 1.0)), "X", "payoff") is None


def test_goals_are_weighed_by_the_preferences_held_for_them(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    context = knowledge.context_named("tictactoe").id  # type: ignore[union-attr]
    winning = knowledge.goal(Goal("payoff", context))
    teaching = knowledge.goal(Goal("taught", context))
    knowledge.prefer(Preference(winning.id, 1.0))
    knowledge.prefer(Preference(teaching.id, 2.0))
    outcome = State.of(payoff=Map.of({"X": 1.0, "O": 0.0}), taught=Map.of({"X": 0.5, "O": 0.0}))

    worth = create_utility().of(knowledge, node, outcomes((outcome, 1.0)), "X", "payoff")

    assert worth == pytest.approx(1.0 * 1.0 + 2.0 * 0.5)


def test_a_goal_nobody_holds_a_preference_for_is_left_out(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    context = knowledge.context_named("tictactoe").id  # type: ignore[union-attr]
    winning = knowledge.goal(Goal("payoff", context))
    knowledge.goal(Goal("taught", context))
    knowledge.prefer(Preference(winning.id, 1.0))
    outcome = State.of(payoff=Map.of({"X": 1.0, "O": 0.0}), taught=Map.of({"X": 0.5, "O": 0.0}))

    assert create_utility().of(knowledge, node, outcomes((outcome, 1.0)), "X", "payoff") == pytest.approx(1.0)


def test_a_role_s_preference_is_taken_over_the_one_held_for_any_role(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    context = knowledge.context_named("tictactoe").id  # type: ignore[union-attr]
    winning = knowledge.goal(Goal("payoff", context))
    knowledge.prefer(Preference(winning.id, 1.0))
    knowledge.prefer(Preference(winning.id, 0.2, role="coach"))

    as_coach = create_utility().of(knowledge, node, outcomes((paid(X=1.0, O=0.0), 1.0)), "X", "payoff", role="coach")

    assert as_coach == pytest.approx(0.2)


def test_continuous_payoffs_are_binned_evenly_each_bin_holding_its_likelihood() -> None:
    winnings = outcomes((paid(gambler=50.0), 0.5), (paid(gambler=150.0), 0.3), (paid(gambler=900.0), 0.2))

    binned = create_even_binner().bins(None, winnings, "gambler", "payoff")

    assert [(round(bin.low), round(bin.high)) for bin in binned] == [(50, 333), (333, 617), (617, 900)]
    assert [bin.likelihood for bin in binned] == [pytest.approx(0.8), 0.0, pytest.approx(0.2)]


def test_payoffs_that_never_vary_give_one_bin_and_none_at_all_give_none() -> None:
    binner = create_even_binner()

    assert binner.bins(None, outcomes((paid(gambler=5.0), 0.4), (paid(gambler=5.0), 0.6)), "gambler", "payoff") == (
        Bin(5.0, 5.0, 1.0),
    )
    assert binner.bins(None, outcomes((paid(gambler=None), 1.0)), "gambler", "payoff") == ()
