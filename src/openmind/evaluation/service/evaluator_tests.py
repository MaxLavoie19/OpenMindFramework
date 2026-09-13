import logging
import math

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.evaluation.factory.evaluator_factory import create_evaluator
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.evaluation.model.rater_agreement import RaterAgreement
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.predictor.model.assign import Assign
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State

pytestmark = pytest.mark.log_level("INFO")


def first_mover_decides() -> Domain:
    """Only A acts: win gives A 1 and B 0, lose the reverse, tie 0.5 each."""
    unset = Equals(StateVariable("payoff", (Constant("A"),)), Constant(None))

    def payoffs(a: float, b: float) -> tuple[Assign, Assign]:
        return (
            Assign(StateVariable("payoff", (Constant("A"),)), Constant(a)),
            Assign(StateVariable("payoff", (Constant("B"),)), Constant(b)),
        )

    return Domain(
        "first mover decides",
        State((("payoff(A)", None), ("payoff(B)", None), ("turn", "A"))),
        Problem(tuple(ActionDefinition(name, (), (unset,)) for name in ("win", "lose", "tie"))),
        TransitionModel(
            (
                Transition("win", (Branch(1.0, payoffs(1.0, 0.0)),)),
                Transition("lose", (Branch(1.0, payoffs(0.0, 1.0)),)),
                Transition("tie", (Branch(1.0, payoffs(0.5, 0.5)),)),
            )
        ),
        Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)")),
    )


class FavorsWin:
    """Rates win above tie above lose."""

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        return tuple({"win": 1.0, "tie": 0.5, "lose": 0.0}[action.name] for action in actions)


class RatesAlike:
    """Rates win and lose the same and knows nothing about tie."""

    def rate(self, state: State, actions: tuple[Action, ...]) -> tuple[float | None, ...]:
        return tuple(None if action.name == "tie" else 0.5 for action in actions)


def test_evaluate_reports_baselines_and_agreement() -> None:
    settings = EvaluationSettings(games=2, iterations=5, positions=10, budgets=(5, 20), seed=1)

    report = create_evaluator().evaluate(
        first_mover_decides(), AgentBuilder().with_exploration(math.sqrt(2)), settings, rules_file="rules.json"
    )

    assert (report.domain, report.rules_file, report.settings) == ("first mover decides", "rules.json", settings)
    assert [(item.opponent, item.games, item.wins + item.draws + item.losses) for item in report.baselines] == [
        ("random", 2, 2),
        ("untrained MCTS", 2, 2),
    ]
    assert report.every_action_optimal == 0
    assert [(item.iterations, item.positions, item.optimal, item.mean_regret) for item in report.agreement] == [
        (5, 1, 1, 0.0),
        (20, 1, 1, 0.0),
    ]
    assert all(0.0 <= item.optimal_visit_share <= 1.0 and item.seconds_per_choice >= 0.0 for item in report.agreement)
    assert (report.unguided_agreement, report.rater) == ((), None)


def test_a_rater_is_compared_with_an_unguided_agent_on_the_same_positions_and_measured_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation")
    settings = EvaluationSettings(games=0, iterations=5, positions=10, budgets=(5, 20), seed=1)
    builder = AgentBuilder().with_exploration(math.sqrt(2)).with_guidance(FavorsWin())

    report = create_evaluator().evaluate(first_mover_decides(), builder, settings, "rules.json", FavorsWin())

    assert [(item.iterations, item.positions) for item in report.agreement] == [(5, 1), (20, 1)]
    assert [(item.iterations, item.positions) for item in report.unguided_agreement] == [(5, 1), (20, 1)]
    assert report.rater == RaterAgreement(positions=1, distinguishing=1, optimal=1.0, mean_regret=0.0)
    assert "Every action is optimal in 0 of 1 positions" in caplog.messages
    assert any(message.startswith("Unguided agreement with perfect play at 20 iterations: ") for message in caplog.messages)
    assert (
        "Rater alone: ratings separate actions in 1 of 1 positions; a top-rated action is optimal in 1.0 of 1; "
        "mean regret 0.0"
    ) in caplog.messages


def test_a_rater_rating_every_action_alike_separates_nothing_and_picks_uniformly() -> None:
    settings = EvaluationSettings(games=0, iterations=5, positions=10, budgets=(5,), seed=1)
    builder = AgentBuilder().with_exploration(math.sqrt(2)).with_guidance(RatesAlike())

    report = create_evaluator().evaluate(first_mover_decides(), builder, settings, "rules.json", RatesAlike())

    assert report.rater is not None
    assert (report.rater.positions, report.rater.distinguishing) == (1, 0)
    assert report.rater.optimal == pytest.approx(1 / 3)
    assert report.rater.mean_regret == pytest.approx(0.5)


def test_all_positions_takes_every_position(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation")
    settings = EvaluationSettings(games=0, iterations=5, positions=None, budgets=(5,), seed=1)

    report = create_evaluator().evaluate(first_mover_decides(), AgentBuilder().with_exploration(math.sqrt(2)), settings)

    assert [(item.iterations, item.positions) for item in report.agreement] == [(5, 1)]
    assert "Every action is optimal in 0 of 1 positions" in caplog.messages


def test_no_positions_skips_agreement_and_the_exact_search(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation")
    settings = EvaluationSettings(games=2, iterations=5, positions=0, budgets=(5, 20), seed=1)

    report = create_evaluator().evaluate(
        first_mover_decides(), AgentBuilder().with_exploration(math.sqrt(2)), settings, "rules.json", FavorsWin()
    )

    assert len(report.baselines) == 2
    assert (report.every_action_optimal, report.agreement, report.unguided_agreement, report.rater) == (0, (), (), None)
    assert "Agreement with perfect play skipped: no positions" in caplog.messages
    assert not any("positions with a legal action" in message for message in caplog.messages)
