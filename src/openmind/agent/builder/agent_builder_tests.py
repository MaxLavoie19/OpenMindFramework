import json

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.prisoners_dilemma_factory import create_prisoners_dilemma_domain
from openmind.agent.factory.rock_paper_scissors_factory import create_rock_paper_scissors_domain
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rbs.service.rule_valuer import RuleValuer
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.mcts.service.semi_determinized_search_tests import Believes, coin_domain
from openmind.timing.model.clock import Clock
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

CENTER = Action("place", (("col", 2), ("row", 2)))


class FavourCenter:
    """A rater rating the center 1.0 and every other action 0.0."""

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        return tuple(1.0 if action == CENTER else 0.0 for action in actions)


@pytest.mark.log_level("INFO")
def test_build_gives_an_agent_that_searches() -> None:
    domain = create_tictactoe_domain()
    builder = StateBuilder()
    x_can_win = {"cell(1,1)": "X", "cell(1,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"}
    for name, value in (dict(domain.initial_state.variables) | x_can_win).items():
        builder.with_variable(name, value)

    agent = AgentBuilder().with_iterations(100).with_exploration(1.4).with_seed(1).build()

    assert agent.choose(domain, builder.build()) == Action("place", (("col", 3), ("row", 1)))


@pytest.mark.log_level("INFO")
def test_build_with_guidance_expands_the_best_rated_action_first() -> None:
    domain = create_tictactoe_domain()

    agent = AgentBuilder().with_iterations(1).with_exploration(1.4).with_seed(1).with_guidance(FavourCenter()).build()

    assert agent.choose(domain, domain.initial_state) == CENTER


class CountingCenter(FavourCenter):
    """FavourCenter, counting how often it is asked to rate."""

    def __init__(self) -> None:
        self.calls = 0

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        self.calls += 1
        return super().rate(state, actions)


@pytest.mark.log_level("INFO")
def test_build_without_guided_rollouts_rates_only_the_tree_nodes() -> None:
    domain = create_tictactoe_domain()
    guided, unguided = CountingCenter(), CountingCenter()

    for rater, rollouts in ((guided, True), (unguided, False)):
        builder = AgentBuilder().with_iterations(1).with_exploration(1.4).with_seed(1).with_guidance(rater)
        builder.with_guided_rollouts(rollouts).build().choose(domain, domain.initial_state)

    # One iteration creates the root and the node after the center; only a guided rollout rates its steps too.
    assert unguided.calls == 2
    assert guided.calls > 2


class ValuesEverything:
    """A valuer giving every position a draw, counting how often it is asked."""

    def __init__(self) -> None:
        self.calls = 0

    def value(self, state: State) -> tuple[float, ...] | None:
        self.calls += 1
        return (0.5, 0.5)


@pytest.mark.log_level("INFO")
def test_build_with_valuation_values_positions_instead_of_playing_them_out() -> None:
    domain = create_tictactoe_domain()
    valuer = ValuesEverything()

    AgentBuilder().with_iterations(3).with_exploration(1.4).with_seed(1).with_valuation(valuer).build().choose(
        domain, domain.initial_state
    )

    # Each iteration reaches one new position still in play, and values it.
    assert valuer.calls == 3


@pytest.mark.log_level("INFO")
def test_build_with_a_rollout_limit_gives_rollouts_still_in_play_the_unfinished_payoff() -> None:
    domain = create_tictactoe_domain()

    agent = AgentBuilder().with_iterations(9).with_exploration(1.4).with_seed(1).with_rollout_limit(0, 0.25).build()

    # Each iteration tries a new first move, and its rollout stops at once.
    assert {(item.visits, item.mean_payoff) for item in agent.search(domain, domain.initial_state).statistics} == {(1, 0.25)}


def x_can_win() -> State:
    """X to act, with X on (1,1) and (1,2) and O on (2,1) and (2,2)."""
    builder = StateBuilder()
    marks = {"cell(1,1)": "X", "cell(1,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"}
    for name, value in (dict(create_tictactoe_domain().initial_state.variables) | marks).items():
        builder.with_variable(name, value)
    return builder.build()


class KnowsNothing:
    """A valuer that can't value any position."""

    def value(self, state: State) -> tuple[float, ...] | None:
        return None


class ValuesTheCorner:
    """A valuer valuing a position 1.0 for X once X holds (3,3), 0.0 otherwise."""

    def value(self, state: State) -> tuple[float, ...] | None:
        return (1.0, 0.0) if dict(state.variables)["cell(3,3)"] == "X" else (0.0, 1.0)


@pytest.mark.log_level("INFO")
@pytest.mark.parametrize("valuer", [None, KnowsNothing()])
def test_build_with_deduction_plays_a_proven_move_without_searching_when_the_rules_have_no_clue(valuer: object) -> None:
    builder = AgentBuilder().with_iterations(50).with_exploration(1.4).with_seed(1).with_deduction(DeductionBudget(1, 30.0))

    result = builder.with_valuation(valuer).build().search(create_tictactoe_domain(), x_can_win())  # type: ignore[arg-type]

    win = Action("place", (("col", 3), ("row", 1)))
    assert (result.player, result.statistics, result.chosen, result.samples) == ("X", (ActionStatistics(win, 1, 1.0),), win, ())


@pytest.mark.log_level("INFO")
def test_build_with_deduction_searches_when_the_rules_tell_the_moves_apart() -> None:
    builder = AgentBuilder().with_iterations(50).with_exploration(1.4).with_seed(1).with_deduction(DeductionBudget(1, 30.0))

    result = builder.with_valuation(ValuesTheCorner()).build().search(create_tictactoe_domain(), x_can_win())

    assert result.samples


@pytest.mark.log_level("INFO")
def test_build_with_deduction_searches_a_domain_with_hidden_information() -> None:
    domain = create_prisoners_dilemma_domain()
    agent = AgentBuilder().with_iterations(20).with_exploration(1.4).with_seed(1).with_deduction(DeductionBudget(2, 30.0)).build()

    assert agent.search(domain, domain.initial_state).samples


@pytest.mark.parametrize("budget", [DeductionBudget(0, 30.0), DeductionBudget(1, 0.0)])
def test_build_rejects_a_deduction_without_plies_or_seconds(budget: DeductionBudget) -> None:
    with pytest.raises(ValueError, match="at least 1 ply"):
        AgentBuilder().with_iterations(1).with_exploration(1.4).with_deduction(budget).build()


@pytest.mark.parametrize(("limit", "payoff", "message"), [(-1, 0.5, "negative"), (5, None, "unfinished payoff")])
def test_build_rejects_a_negative_rollout_limit_or_one_without_an_unfinished_payoff(
    limit: int, payoff: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentBuilder().with_iterations(1).with_exploration(1.4).with_rollout_limit(limit, payoff).build()


def test_build_rejects_negative_rollout_actions() -> None:
    with pytest.raises(ValueError, match="-1"):
        AgentBuilder().with_iterations(1).with_exploration(1.4).with_rollout_actions(-1).build()


def test_build_rejects_missing_settings() -> None:
    with pytest.raises(ValueError, match="iterations, a time budget estimator or both"):
        AgentBuilder().with_exploration(1.4).build()
    with pytest.raises(ValueError, match="exploration"):
        AgentBuilder().with_iterations(10).build()


def test_build_with_a_time_budget_estimator_gives_an_agent_searching_on_time_alone() -> None:
    domain = create_tictactoe_domain()
    agent = AgentBuilder().with_exploration(1.4).with_seed(1).with_time_budget_estimator(PlainTimeBudgetEstimator(30)).build()

    result = agent.search(domain, domain.initial_state, clock=Clock(0.3))

    assert result.iterations >= 1


def test_build_rejects_fewer_than_one_iteration() -> None:
    with pytest.raises(ValueError, match="0"):
        AgentBuilder().with_iterations(0).with_exploration(1.4).build()


def test_an_agent_with_a_theory_of_mind_searches_once_per_hypothesis_in_a_domain_with_an_observation() -> None:
    domain = coin_domain("tails", 0.5)

    result = AgentBuilder().with_iterations(40).with_exploration(1.4).with_seed(1).with_theory_of_mind(Believes(0.8)).build().search(
        domain, domain.initial_state
    )

    assert result.chosen == Action("guess", (("side", "heads"),))
    assert [hypothesis.probability for hypothesis in result.hypotheses] == [0.8, pytest.approx(0.2)]


def test_an_agent_with_a_theory_of_mind_searches_plainly_where_every_player_sees_everything() -> None:
    domain = create_tictactoe_domain()

    result = AgentBuilder().with_iterations(20).with_exploration(1.4).with_seed(1).with_theory_of_mind().build().search(
        domain, domain.initial_state
    )

    assert result.hypotheses == ()


def throw(shape: str) -> Action:
    return Action("throw", (("shape", shape),))


class PredictsRock:
    """A theory of mind predicting the other player throws rock 0.6 of the time."""

    def hypotheses(self, domain: object, observed: State, player: str) -> tuple[()]:
        return ()

    def strategy(self, domain: object, state: State, player: str, other: str) -> tuple[tuple[Action, float], ...]:
        return (throw("rock"), 0.6), (throw("paper"), 0.2), (throw("scissors"), 0.2)


def test_an_agent_acting_at_once_searches_for_its_player_against_the_strategy_its_theory_predicts() -> None:
    domain = create_rock_paper_scissors_domain()
    agent = AgentBuilder().with_iterations(2000).with_exploration(1.4).with_seed(1).with_theory_of_mind(PredictsRock()).build()  # type: ignore[arg-type]

    result = agent.search(domain, domain.initial_state, "A")

    assert result.player == "A" and dict(result.strategy)[throw("paper")] > 0.5
    assert agent.choose(domain, domain.initial_state, "B") in (throw("rock"), throw("paper"), throw("scissors"))


def test_a_description_is_the_same_for_the_same_settings_whatever_the_seed_and_changes_with_any_setting() -> None:
    def builder() -> AgentBuilder:
        return AgentBuilder().with_iterations(100).with_exploration(1.4).with_rollout_limit(100, 0.5)

    same = builder().with_seed(1).describe("arm"), builder().with_seed(2).describe("arm")
    other = builder().with_iterations(101).describe("arm")

    assert same[0] == same[1] and same[0].id == same[1].id
    assert other.id != same[0].id
    assert json.loads(same[0].text)["rollout_limit"] == 100


def test_a_rule_valuer_is_described_by_its_value_base_and_a_part_that_can_t_describe_itself_is_marked() -> None:
    domain = create_tictactoe_domain()
    base = ValueBase("tictactoe", 0.1, 0.0, 1.0, (ValueRule(PythonRule("1.0"), 0.5),))
    valuer = RuleValuer(base, domain, RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())), ConsequenceLibraryBuilder().build())
    estimator = PlainTimeBudgetEstimator(30)

    text = json.loads(
        AgentBuilder().with_iterations(10).with_exploration(1.4).with_valuation(valuer).with_guidance(FavourCenter()).with_time_budget_estimator(estimator).describe("rules").text
    )
    changed = AgentBuilder().with_iterations(10).with_exploration(1.4).with_valuation(
        RuleValuer(ValueBase("tictactoe", 0.2, 0.0, 1.0, base.rules), domain, RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())), ConsequenceLibraryBuilder().build())
    )

    assert ValueBaseJsonMapper().from_json(json.dumps(text["valuation"])) == base
    assert text["guidance"] == {"class": "openmind.agent.builder.agent_builder_tests.FavourCenter", "not rebuildable": True}
    assert text["time_budget_estimator"] == {"rule": "plain", "expected_steps": 30, "reserve_seconds": 0.0}
    assert changed.describe("rules").id != AgentBuilder().with_iterations(10).with_exploration(1.4).with_valuation(valuer).describe("rules").id
