import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.rbs.model.induction_settings import InductionSettings
from openmind.training.factory.training_factory import create_distiller
from openmind.training.model.distillation_settings import DistillationSettings

pytestmark = pytest.mark.log_level("INFO")


def test_distill_induces_rules_and_measures_them_on_held_out_games() -> None:
    settings = DistillationSettings(
        games=2,
        held_out_games=1,
        iterations=20,
        seed=1,
        induction=InductionSettings(min_visits=1, max_conditions=1, min_rule_visits=5, min_gain=0.05),
    )

    result = create_distiller().distill(create_tictactoe_domain(), AgentBuilder().with_exploration(1.4), settings)

    assert result.rule_base.domain == "tictactoe"
    assert result.rule_base.rules[0].action == "place"
    assert result.rule_base.rules[0].conditions == ()
    assert result.training_samples > 0
    assert result.held_out_samples > 0
    assert isinstance(result.rating_error, float)
    assert result.mean_conditions >= 0.0
