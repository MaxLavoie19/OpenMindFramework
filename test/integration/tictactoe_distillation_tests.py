import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.rbs.constant.generation_constant import DEFAULT_PERMUTATIONS
from openmind.rbs.model.generation_settings import GenerationSettings
from openmind.training.factory.training_factory import create_distiller
from openmind.training.model.distillation_settings import DistillationSettings

pytestmark = pytest.mark.log_level("INFO")


def test_distillation_tests_hypotheses_on_held_out_games_and_keeps_the_validated_ones_no_simpler_rule_covers() -> None:
    generation = GenerationSettings(
        min_visits=5,
        max_conditions=2,
        min_rule_visits=50,
        min_gain=0.05,
        confidence=0.95,
        beam_width=20,
        max_offset=2,
        solo_limit=2,
        patterns=200,
        false_discovery_rate=0.05,
        permutations=DEFAULT_PERMUTATIONS,
    )
    settings = DistillationSettings(games=4, held_out_games=4, iterations=50, seed=1, generation=generation)

    result = create_distiller().distill(create_tictactoe_domain(), AgentBuilder().with_exploration(EXPLORATION), settings)

    assert result.hypotheses
    assert all(0.0 <= test.p_value <= test.q_value <= 1.0 for test in result.hypotheses)
    kept = {rule.conditions for rule in result.rule_base.rules if rule.conditions}
    covered = {coverage.rule.conditions for coverage in result.covered}
    assert kept.isdisjoint(covered)
    assert kept | covered == {test.conditions for test in result.hypotheses if test.validated}
