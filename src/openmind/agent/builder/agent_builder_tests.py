from collections.abc import Callable
import json

from pathlib import Path

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import create_rule_based_system
from openmind.rule.model.python_rule import PythonRule
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.model.clock import Clock
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.world.model.action import Action
from openmind.world.model.state import State

CENTER = Action("place", (("col", 2), ("row", 2)))

type Game = Callable[[str], RuleBasedSystem]


class FavourCenter:
    """A rater rating the center 1.0 and every other action 0.0."""

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        return tuple(1.0 if action == CENTER else 0.0 for action in actions)


@pytest.mark.log_level("INFO")
def test_build_gives_an_agent_that_searches(game: Game) -> None:
    rbs = game("tictactoe")
    agent = AgentBuilder().with_iterations(100).with_exploration(1.4).with_seed(1).build()

    assert agent.choose(rbs, x_can_win(rbs)) == Action("place", (("col", 3), ("row", 1)))


@pytest.mark.log_level("INFO")
def test_build_with_guidance_expands_the_best_rated_action_first(game: Game) -> None:
    rbs = game("tictactoe")

    agent = AgentBuilder().with_iterations(1).with_exploration(1.4).with_seed(1).with_guidance(FavourCenter()).build()

    assert agent.choose(rbs, rbs.start()) == CENTER


class CountingCenter(FavourCenter):
    """FavourCenter, counting how often it is asked to rate."""

    def __init__(self) -> None:
        self.calls = 0

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        self.calls += 1
        return super().rate(state, actions)


@pytest.mark.log_level("INFO")
def test_build_without_guided_rollouts_rates_only_the_tree_nodes(game: Game) -> None:
    rbs = game("tictactoe")
    guided, unguided = CountingCenter(), CountingCenter()

    for rater, rollouts in ((guided, True), (unguided, False)):
        builder = AgentBuilder().with_iterations(1).with_exploration(1.4).with_seed(1).with_guidance(rater)
        builder.with_guided_rollouts(rollouts).build().choose(rbs, rbs.start())

    # One iteration creates the root and the node after the center; only a guided rollout rates its steps too.
    assert unguided.calls == 2
    assert guided.calls > 2


class ValuesEverything:
    """A valuer giving every position a draw, counting how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    def values(self, state: State) -> tuple[float, ...] | None:
        self.calls += 1
        return (0.5, 0.5)


@pytest.mark.log_level("INFO")
def test_build_with_valuation_values_positions_instead_of_playing_them_out(game: Game) -> None:
    rbs = game("tictactoe")
    valuer = ValuesEverything()

    AgentBuilder().with_iterations(3).with_exploration(1.4).with_seed(1).with_valuation(valuer).build().choose(
        rbs, rbs.start()
    )

    # Each iteration reaches one new position still in play, and values it.
    assert valuer.calls == 3


@pytest.mark.log_level("INFO")
def test_build_with_a_rollout_limit_gives_rollouts_still_in_play_the_unfinished_payoff(game: Game) -> None:
    rbs = game("tictactoe")

    agent = AgentBuilder().with_iterations(9).with_exploration(1.4).with_seed(1).with_rollout_limit(0, 0.25).build()

    # Each iteration tries a new first move, and its rollout stops at once.
    assert {(item.visits, item.mean_payoff) for item in agent.search(rbs, rbs.start()).statistics} == {(1, 0.25)}


type Game = Callable[[str], RuleBasedSystem]


def x_can_win(rbs: RuleBasedSystem) -> State:
    """X to act, with X on (1,1) and (1,2) and O on (2,1) and (2,2)."""
    start = rbs.start()
    cell = start.model("cell")
    for where, mark in (((1, 1), "X"), ((1, 2), "X"), ((2, 1), "O"), ((2, 2), "O")):
        cell = cell.placed(where, mark)
    return start.with_model("cell", cell)


class KnowsNothing:
    """A valuer that can't value any position."""

    def values(self, state: State) -> tuple[float, ...] | None:
        return None


class ValuesTheCorner:
    """A valuer valuing a position 1.0 for X once X holds (3,3), 0.0 otherwise."""

    def values(self, state: State) -> tuple[float, ...] | None:
        return (1.0, 0.0) if state.model("cell")[3, 3] == "X" else (0.0, 1.0)


@pytest.mark.log_level("INFO")
@pytest.mark.parametrize("valuer", [None, KnowsNothing()])
def test_build_with_deduction_plays_a_proven_move_without_searching_when_the_rules_have_no_clue(game: Game, valuer: object) -> None:
    builder = AgentBuilder().with_iterations(50).with_exploration(1.4).with_seed(1).with_deduction(DeductionBudget(1, 30.0))

    rbs = game("tictactoe")

    result = builder.with_valuation(valuer).build().search(rbs, x_can_win(rbs))  # type: ignore[arg-type]

    win = Action("place", (("col", 3), ("row", 1)))
    assert (result.player, result.statistics, result.chosen, result.samples) == ("X", (ActionStatistics(win, 1, 1.0),), win, ())


@pytest.mark.log_level("INFO")
def test_build_with_deduction_searches_when_the_rules_tell_the_moves_apart(game: Game) -> None:
    builder = AgentBuilder().with_iterations(50).with_exploration(1.4).with_seed(1).with_deduction(DeductionBudget(1, 30.0))

    rbs = game("tictactoe")

    result = builder.with_valuation(ValuesTheCorner()).build().search(rbs, x_can_win(rbs))

    assert result.samples


@pytest.mark.log_level("INFO")
def test_build_with_deduction_searches_a_domain_with_hidden_information(game: Game) -> None:
    rbs = game("prisonersdilemma")
    agent = AgentBuilder().with_iterations(20).with_exploration(1.4).with_seed(1).with_deduction(DeductionBudget(2, 30.0)).build()

    assert agent.search(rbs, rbs.start()).samples


@pytest.mark.parametrize("budget", [DeductionBudget(0, 30.0), DeductionBudget(1, 0.0)])
def test_build_rejects_a_deduction_without_plies_or_seconds(game: Game, budget: DeductionBudget) -> None:
    with pytest.raises(ValueError, match="at least 1 ply"):
        AgentBuilder().with_iterations(1).with_exploration(1.4).with_deduction(budget).build()


@pytest.mark.parametrize(("limit", "payoff", "message"), [(-1, 0.5, "negative"), (5, None, "unfinished payoff")])
def test_build_rejects_a_negative_rollout_limit_or_one_without_an_unfinished_payoff(game: Game, 
    limit: int, payoff: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentBuilder().with_iterations(1).with_exploration(1.4).with_rollout_limit(limit, payoff).build()


def test_build_rejects_negative_rollout_actions(game: Game) -> None:
    with pytest.raises(ValueError, match="-1"):
        AgentBuilder().with_iterations(1).with_exploration(1.4).with_rollout_actions(-1).build()


def test_build_rejects_missing_settings(game: Game) -> None:
    with pytest.raises(ValueError, match="iterations, a time budget estimator or both"):
        AgentBuilder().with_exploration(1.4).build()
    with pytest.raises(ValueError, match="exploration"):
        AgentBuilder().with_iterations(10).build()


def test_build_with_a_time_budget_estimator_gives_an_agent_searching_on_time_alone(game: Game) -> None:
    rbs = game("tictactoe")
    agent = AgentBuilder().with_exploration(1.4).with_seed(1).with_time_budget_estimator(PlainTimeBudgetEstimator(30)).build()

    result = agent.search(rbs, rbs.start(), clock=Clock(0.3))

    assert result.iterations >= 1


def test_build_rejects_fewer_than_one_iteration(game: Game) -> None:
    with pytest.raises(ValueError, match="0"):
        AgentBuilder().with_iterations(0).with_exploration(1.4).build()


def test_a_description_is_the_same_for_the_same_settings_whatever_the_seed_and_changes_with_any_setting(game: Game) -> None:
    def builder() -> AgentBuilder:
        return AgentBuilder().with_iterations(100).with_exploration(1.4).with_rollout_limit(100, 0.5)

    same = builder().with_seed(1).describe("arm"), builder().with_seed(2).describe("arm")
    other = builder().with_iterations(101).describe("arm")

    assert same[0] == same[1] and same[0].id == same[1].id
    assert other.id != same[0].id
    assert json.loads(same[0].text)["rollout_limit"] == 100


def test_a_valuer_is_described_by_its_rules_and_a_part_that_can_t_describe_itself_is_marked(
    game: Game, knowledge: KnowledgeBase
) -> None:
    rbs = game("tictactoe")
    RuleDeclarer(knowledge, rbs.context).position("a constant", PythonRule("1.0"), 0.5)
    valuer = create_rule_based_system(knowledge, rbs.context)
    estimator = PlainTimeBudgetEstimator(30)

    text = json.loads(
        AgentBuilder().with_iterations(10).with_exploration(1.4).with_valuation(valuer).with_guidance(FavourCenter()).with_time_budget_estimator(estimator).describe("rules").text
    )
    RuleDeclarer(knowledge, rbs.context).position("a constant", PythonRule("1.0"), 0.9)
    changed = AgentBuilder().with_iterations(10).with_exploration(1.4).with_valuation(
        create_rule_based_system(knowledge, rbs.context)
    )

    assert text["valuation"] == {"context": "tictactoe", "position": [["a constant", 0.5]], "move": []}
    assert text["guidance"] == {"class": "openmind.agent.builder.agent_builder_tests.FavourCenter", "not rebuildable": True}
    assert text["time_budget_estimator"] == {"rule": "plain", "expected_steps": 30, "reserve_seconds": 0.0}
    assert changed.describe("rules").id != AgentBuilder().with_iterations(10).with_exploration(1.4).with_valuation(valuer).describe("rules").id
