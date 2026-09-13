import logging
import math

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.domain import Domain
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.evaluation.factory.evaluator_factory import create_evaluator
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.predictor.model.assign import Assign
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.world.model.players import Players
from openmind.world.model.state import State


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


@pytest.mark.log_level("INFO")
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
    assert [(item.iterations, item.positions, item.optimal) for item in report.agreement] == [(5, 1, 1), (20, 1, 1)]
    assert all(item.seconds_per_choice >= 0.0 for item in report.agreement)


@pytest.mark.log_level("INFO")
def test_no_positions_skips_agreement_and_the_exact_search(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.evaluation")
    settings = EvaluationSettings(games=2, iterations=5, positions=0, budgets=(5, 20), seed=1)

    report = create_evaluator().evaluate(first_mover_decides(), AgentBuilder().with_exploration(math.sqrt(2)), settings)

    assert len(report.baselines) == 2
    assert report.agreement == ()
    assert "Agreement with perfect play skipped: no positions" in caplog.messages
    assert not any("positions with a legal action" in message for message in caplog.messages)
