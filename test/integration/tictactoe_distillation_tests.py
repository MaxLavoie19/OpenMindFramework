import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.evaluation.factory.evaluator_factory import create_evaluator
from openmind.evaluation.model.evaluation_settings import EvaluationSettings
from openmind.rbs.factory.rbs_factory import create_rule_rater
from openmind.rbs.model.induction_settings import InductionSettings
from openmind.training.factory.training_factory import create_distiller
from openmind.training.model.distillation_settings import DistillationSettings

pytestmark = pytest.mark.log_level("INFO")


def test_distilled_rules_guide_the_agent_to_at_least_the_untrained_agreement_at_20_iterations() -> None:
    domain = create_tictactoe_domain()
    settings = DistillationSettings(
        games=10,
        held_out_games=2,
        iterations=100,
        seed=1,
        induction=InductionSettings(min_visits=5, max_conditions=2, min_rule_visits=50, min_gain=0.05),
    )

    result = create_distiller().distill(domain, AgentBuilder().with_exploration(EXPLORATION), settings)

    assert any(rule.action == "place" and rule.conditions for rule in result.rule_base.rules)
    evaluation = EvaluationSettings(games=0, iterations=20, positions=50, budgets=(20,), seed=1)
    untrained = create_evaluator().evaluate(domain, AgentBuilder().with_exploration(EXPLORATION), evaluation)
    guided_builder = AgentBuilder().with_exploration(EXPLORATION).with_guidance(create_rule_rater(result.rule_base))
    guided = create_evaluator().evaluate(domain, guided_builder, evaluation)
    assert guided.agreement[0].optimal >= untrained.agreement[0].optimal
