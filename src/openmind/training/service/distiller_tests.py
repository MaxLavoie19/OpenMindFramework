from dataclasses import replace

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.rbs.service.primitive_generator_tests import SETTINGS
from openmind.training.factory.training_factory import create_distiller
from openmind.training.model.distillation_settings import DistillationSettings

pytestmark = pytest.mark.log_level("INFO")


def test_distill_generates_rules_validates_them_and_measures_them_on_held_out_games() -> None:
    settings = DistillationSettings(
        games=2,
        held_out_games=2,
        iterations=20,
        seed=1,
        generation=replace(SETTINGS, max_conditions=1, min_rule_visits=5, solo_limit=1, permutations=200),
    )

    result = create_distiller().distill(create_tictactoe_domain(), AgentBuilder().with_exploration(1.4), settings)

    assert result.rule_base.domain == "tictactoe"
    assert (result.rule_base.rules[0].action, result.rule_base.rules[0].conditions) == ("place", ())
    assert result.training_samples > 0
    assert result.held_out_samples > 0
    assert isinstance(result.rating_error, float)
    assert result.mean_conditions >= 0.0
    assert {rule.conditions for rule in result.rule_base.rules[1:]} | {
        coverage.rule.conditions for coverage in result.covered
    } == {test.conditions for test in result.hypotheses if test.validated}
