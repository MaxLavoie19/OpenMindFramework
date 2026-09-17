import math
import time

import pytest

from openmind.agent.factory.rock_paper_scissors_factory import create_rock_paper_scissors_domain
from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.model.domain import Domain
from openmind.agent.service.deduction_fallback import DeductionFallback
from openmind.agent.service.move_planner import MovePlanner
from openmind.agent.service.one_ply_chooser import OnePlyChooser
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.service.valuation_prior_tests import CenterValued
from openmind.parallel.service.task_runner import TaskRunner
from openmind.timing.model.time_control import TimeControl
from openmind.agent.service.agent import Agent
from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.semi_determinized_search import SemiDeterminizedSearch
from openmind.mcts.service.semi_determinized_search_tests import Believes, coin_domain
from openmind.mcts.service.tree_search import TreeSearch
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.timing.model.clock import Clock
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


def win_or_lose() -> Domain:
    no_payoff = PythonRule("payoff is None")
    return Domain(
        "win or lose",
        State((("payoff", None), ("turn", "me"))),
        Problem((ActionDefinition("lose", (), (no_payoff,)), ActionDefinition("win", (), (no_payoff,)))),
        TransitionModel(
            (
                Transition("lose", (Branch(1.0, PythonRule("payoff = 0.0")),)),
                Transition("win", (Branch(1.0, PythonRule("payoff = 1.0")),)),
            )
        ),
        Players(("me",), "turn", ("payoff",)),
    )


def new_agent() -> Agent:
    tree_search = TreeSearch(create_solver(), create_predictor(), StateReader(), ActionTextMapper())
    return Agent(tree_search, SearchSettings(50, math.sqrt(2), 1))


def test_choose_returns_the_most_visited_action() -> None:
    domain = win_or_lose()

    assert new_agent().choose(domain, domain.initial_state) == Action("win", ())


def test_search_returns_the_statistics_and_the_tree_samples() -> None:
    domain = win_or_lose()

    result = new_agent().search(domain, domain.initial_state)

    assert result.chosen == Action("win", ())
    assert {sample.action.name for sample in result.samples} == {"lose", "win"}


def timed_agent(iterations: int | None, estimator: bool = True, theory: bool = False) -> Agent:
    """An agent whose search reads a time source moving on by a second a reading, estimating a step's budget as all the
    time left."""
    tree_search = TreeSearch(create_solver(), create_predictor(), StateReader(), ActionTextMapper(), time_source=Ticking())
    semi_determinized = (
        SemiDeterminizedSearch(tree_search, create_state_observer(), StateReader(), ActionTextMapper()) if theory else None
    )
    return Agent(
        tree_search,
        SearchSettings(iterations, math.sqrt(2), 1),
        semi_determinized_search=semi_determinized,
        theory=Believes(0.5) if theory else None,
        estimator=PlainTimeBudgetEstimator(1) if estimator else None,
    )


def test_an_agent_given_a_clock_searches_for_the_step_s_budget() -> None:
    domain = win_or_lose()

    result = timed_agent(None).search(domain, domain.initial_state, clock=Clock(5.0), steps_played=3)

    assert result.iterations == 5
    assert result.chosen == Action("win", ())


def test_on_a_clock_the_budget_replaces_the_iterations_an_agent_was_built_with() -> None:
    domain = win_or_lose()

    assert timed_agent(3).search(domain, domain.initial_state, clock=Clock(5.0)).iterations == 5
    assert timed_agent(3).search(domain, domain.initial_state).iterations == 3


def test_an_agent_without_a_clock_searches_its_iterations() -> None:
    domain = win_or_lose()

    assert timed_agent(50).search(domain, domain.initial_state).iterations == 50


def test_an_agent_given_a_clock_without_an_estimator_raises() -> None:
    domain = win_or_lose()

    with pytest.raises(ValueError, match="time budget estimator"):
        timed_agent(50, estimator=False).choose(domain, domain.initial_state, clock=Clock(5.0))


def test_an_agent_built_without_iterations_needs_a_clock() -> None:
    domain = win_or_lose()

    with pytest.raises(ValueError, match="needs a clock"):
        timed_agent(None).choose(domain, domain.initial_state)


def test_a_semi_determinized_agent_shares_the_step_s_budget_between_hypotheses() -> None:
    domain = coin_domain("tails", 0.5)

    result = timed_agent(None, theory=True).search(domain, domain.initial_state, clock=Clock(10.0))

    assert len(result.hypotheses) == 2
    assert result.iterations == 6


def test_an_agent_acting_at_once_searches_for_the_step_s_budget() -> None:
    domain = create_rock_paper_scissors_domain()

    assert timed_agent(None).search(domain, domain.initial_state, "A", Clock(5.0)).iterations == 5


def clocked_agent(fallback_cost: float | None) -> Agent:
    """An agent on a time source moving on 10 ms a reading, with a deduction fallback whose cost per legal move the
    planner has seen when given, estimating a move's budget as the time left above a 1 s reserve."""
    solver, predictor, state_reader = create_solver(), create_predictor(), StateReader()
    tree_search = TreeSearch(solver, predictor, state_reader, ActionTextMapper(), time_source=Ticking(0.01))
    planner = MovePlanner()
    if fallback_cost is not None:
        planner.observe("fallback", fallback_cost * 9, 9)
    fallback = DeductionFallback(
        PositionDeducer(solver, predictor, state_reader, ActionTextMapper()), solver, predictor, state_reader, DeductionBudget(1, 5.0)
    )
    return Agent(
        tree_search,
        SearchSettings(None, math.sqrt(2), 1),
        valuation=LeafValuation(CenterValued()),
        fallback=fallback,
        estimator=PlainTimeBudgetEstimator(1, 1.0),
        one_ply=OnePlyChooser(solver, predictor, state_reader),
        planner=planner,
    )


def test_an_agent_on_a_clock_searches_after_its_fallback_when_it_leaves_time_and_alone_otherwise(caplog: pytest.LogCaptureFixture) -> None:
    domain = create_tictactoe_domain()

    # 9 legal moves and a 2 s budget: a fallback at 0.1 s a move leaves time, one at 0.3 s doesn't
    full = clocked_agent(0.1).search(domain, domain.initial_state, clock=Clock(3.0))
    alone = clocked_agent(0.3).search(domain, domain.initial_state, clock=Clock(3.0))
    random_move = clocked_agent(0.1).search(domain, domain.initial_state, clock=Clock(0.9))

    assert (full.option, full.budget, full.iterations > 0) == ("fallback and search", 2.0, True)
    assert (alone.option, alone.iterations > 0) == ("search", True)
    assert (random_move.option, random_move.budget) == ("random", 0.0)
    assert any(message.startswith("X plays by random within a 0.000 second budget: ") for message in caplog.messages)


def test_a_fallback_cut_short_by_the_deadline_still_leaves_the_move_a_search() -> None:
    domain = create_tictactoe_domain()
    agent = clocked_agent(None)

    result = agent.search(domain, domain.initial_state, clock=Clock(1.05))

    assert result.option == "fallback and search" and result.iterations >= 1


def test_an_agent_on_a_tight_clock_with_slow_valuations_never_runs_out_of_time() -> None:
    domain = create_tictactoe_domain()

    def slow_agent(seed: int) -> Agent:
        return (
            AgentBuilder()
            .with_exploration(1.4)
            .with_iterations(1000)
            .with_seed(seed)
            .with_valuation(SlowCenterValued())
            .with_deduction(DeductionBudget(2, 1.0))
            .with_time_budget_estimator(PlainTimeBudgetEstimator(3, 0.2))
            .build()
        )

    runner = MatchRunner(create_solver(), create_predictor(), StateReader(), TaskRunner(1))
    games = [runner.play_game(domain, slow_agent, slow_agent, seat, seed, seed, TimeControl(2.0)) for seat, seed in ((0, 1), (1, 2))]

    assert [game.flagged for game in games] == [None, None]


class SlowCenterValued(CenterValued):
    """Values as CenterValued, taking 20 milliseconds a position."""

    def value(self, state: State) -> tuple[float, ...] | None:
        time.sleep(0.02)
        return super().value(state)


def test_where_the_planner_doesn_t_choose_a_budget_of_0_searches_a_single_iteration() -> None:
    domain = create_rock_paper_scissors_domain()

    result = timed_agent(None).search(domain, domain.initial_state, "A", Clock(0.0))

    assert (result.iterations, result.budget) == (1, 0.0)
