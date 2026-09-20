import math
from collections.abc import Callable

import pytest

from openmind.heuristic.model.node import Node
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.factory.search_factory import create_monte_carlo_tree_search
from openmind.search.model.guidance import Guidance
from openmind.search.model.search_settings import SearchSettings
from openmind.world.model.action import Action
from openmind.world.model.state import State

type Game = Callable[[str], RuleBasedGame]


class Values:
    """A position value model of the test's own: what it was told each player is worth, whatever the position."""

    def __init__(self, values: tuple[float, ...] | None) -> None:
        self.values = values
        self.readings = 0

    def values_of(self, model: object, node: Node) -> tuple[float, ...] | None:
        self.readings += 1
        return self.values


class Rates:
    """A move value model of the test's own: the actions it was told to favour, rated 1, the rest rated 0."""

    def __init__(self, favoured: tuple[str, ...] = ()) -> None:
        self.favoured = favoured

    def rate(self, model: object, node: Node, actions: tuple[Action, ...], player: str) -> tuple[float | None, ...]:
        return tuple(1.0 if str(dict(action.parameters)) in self.favoured else 0.0 for action in actions)


def valuer(values: tuple[float, ...] | None) -> tuple[object, object]:
    held = Values(values)
    return held, type("Valuer", (), {"values": staticmethod(held.values_of)})()


def played(game: Game, name: str = "tictactoe") -> tuple[RuleBasedGame, Node]:
    playing = game(name)
    return playing, playing.node(playing.start())


def test_a_search_told_nothing_about_how_much_to_explore_says_so(game: Game, knowledge: KnowledgeBase) -> None:
    _, node = played(game)

    with pytest.raises(ValueError, match="how many nodes"):
        create_monte_carlo_tree_search().plan(None, knowledge, node, Guidance("X"), SearchSettings())


def test_what_it_played_is_what_it_explored_each_as_often_as_it_looked_at_it(
    game: Game, knowledge: KnowledgeBase
) -> None:
    playing, node = played(game)

    strategy = create_monte_carlo_tree_search().plan(
        None, knowledge, node, Guidance("X"), SearchSettings(nodes=40, seed=1)
    )

    assert strategy is not None
    distribution = strategy.at(node.state)
    assert math.isclose(math.fsum(probability for _, probability in distribution), 1.0)
    assert strategy.chosen(node.state) in playing.actions(node.state, player="X")


def test_a_move_it_never_looked_at_is_not_one_it_recommends(game: Game, knowledge: KnowledgeBase) -> None:
    """Nine moves and five nodes: exploring is what says a move is worth considering, so what it didn't look at
    carries none of the strategy."""
    playing, node = played(game)

    strategy = create_monte_carlo_tree_search().plan(None, knowledge, node, Guidance("X"), SearchSettings(nodes=5, seed=1))

    assert strategy is not None
    assert 0 < len(strategy.at(node.state)) < len(playing.actions(node.state, player="X"))


def test_competitive_play_takes_the_best_move_alone_and_bootstrapping_stays_permissive(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """Exploring a move isn't the same as playing it: at a temperature of 1 the strategy is what was explored, and at
    0 it is the best of it and nothing else."""
    _, node = played(game)

    permissive = create_monte_carlo_tree_search().plan(
        None, knowledge, node, Guidance("X"), SearchSettings(nodes=60, seed=1, temperature=1.0)
    )
    competitive = create_monte_carlo_tree_search().plan(
        None, knowledge, node, Guidance("X"), SearchSettings(nodes=60, seed=1, temperature=0.0)
    )

    assert permissive is not None and competitive is not None
    explored = dict(permissive.at(node.state))
    taken = dict(competitive.at(node.state))
    assert len(explored) > len(taken)  # it explored more moves than it would play
    best = max(explored.values())
    assert all(math.isclose(explored[action], best) for action in taken)  # only the best of what it explored
    assert math.isclose(math.fsum(taken.values()), 1.0)


def test_a_search_with_no_heuristic_learns_from_the_wins_it_reaches(game: Game, knowledge: KnowledgeBase) -> None:
    """Tic-tac-toe is small enough to stumble on wins, so a search with nothing to value a position with still plays:
    what it learns is what the finished positions paid."""
    playing, node = played(game)

    strategy = create_monte_carlo_tree_search().plan(
        None, knowledge, node, Guidance("X"), SearchSettings(nodes=300, seed=3)
    )

    assert strategy is not None and strategy.chosen(node.state) in playing.actions(node.state, player="X")
    assert len(strategy.at(node.state)) == 9  # every move was tried before one was recommended


def test_the_position_value_model_is_what_a_leaf_is_worth(game: Game, knowledge: KnowledgeBase) -> None:
    _, node = played(game)
    model, service = valuer((0.9, 0.1))

    strategy = create_monte_carlo_tree_search().plan(
        None, knowledge, node, Guidance("X", position_value=(model, service)), SearchSettings(nodes=100, seed=1)
    )

    assert strategy is not None
    assert model.readings > 0  # type: ignore[attr-defined] - the heuristic is what the leaves were worth


def test_what_a_move_is_rated_leads_the_search_before_anything_is_explored(game: Game, knowledge: KnowledgeBase) -> None:
    playing, node = played(game)
    centre = str({"col": 2, "row": 2})
    rater = Rates((centre,))

    strategy = create_monte_carlo_tree_search().plan(
        None,
        knowledge,
        node,
        Guidance("X", move_value=(None, rater), position_value=valuer((0.5, 0.5))),
        SearchSettings(nodes=100, seed=1),
    )

    assert strategy is not None
    distribution = dict((str(dict(action.parameters)), probability) for action, probability in strategy.at(node.state))
    assert distribution[centre] == max(distribution.values())


def test_the_tree_is_kept_so_the_next_move_carries_on_from_what_was_found(game: Game, knowledge: KnowledgeBase) -> None:
    playing, node = played(game)
    search = create_monte_carlo_tree_search()
    settings = SearchSettings(nodes=40, seed=1)

    first = search.plan(None, knowledge, node, Guidance("X"), settings)
    assert first is not None
    played_on, _ = playing.outcomes(node.state, first.chosen(node.state)).outcomes[0]  # type: ignore[arg-type]
    second = search.plan(None, knowledge, playing.node(played_on), Guidance("O"), settings)

    assert second is not None
    assert second.chosen(played_on) in playing.actions(played_on, player="O")


def test_players_acting_at_once_settle_on_a_mixed_strategy(game: Game, knowledge: KnowledgeBase) -> None:
    """Rock paper scissors has one answer, throwing each shape a third of the time, which regret matching heads
    toward."""
    playing, node = played(game, "rockpaperscissors")

    strategy = create_monte_carlo_tree_search().plan(
        None, knowledge, node, Guidance("A"), SearchSettings(nodes=300, seed=1)
    )

    assert strategy is not None
    probabilities = [probability for _, probability in strategy.at(node.state)]
    assert len(probabilities) == 3
    assert all(0.2 < probability < 0.5 for probability in probabilities)
