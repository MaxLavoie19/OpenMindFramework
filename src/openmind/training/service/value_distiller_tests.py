import logging

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.factory.training_factory import create_value_distiller
from openmind.training.model.value_distillation_settings import ValueDistillationSettings

pytestmark = pytest.mark.log_level("INFO")

VALUES = ValueSettings(pair_pool=5, cuts=3, solo_limit=1, prices=(0.1, 0.01), max_steps=200, tolerance=1e-6)


@pytest.mark.parametrize(("target", "rows_per_position"), [("outcome", 2), ("search", 1)])
def test_distill_fits_value_rules_on_self_play_positions_and_measures_them_on_held_out_games(
    target: str, rows_per_position: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    settings = ValueDistillationSettings(games=3, held_out_games=2, iterations=20, seed=1, target=target, values=VALUES)

    result = create_value_distiller().distill(create_tictactoe_domain(), AgentBuilder().with_exploration(1.4), settings)

    assert result.value_base.domain == "tictactoe"
    assert [fit.price for fit in result.fits] == [0.1, 0.01]
    assert result.training_rows % rows_per_position == 0 and result.training_rows >= 3 * 5 * rows_per_position
    assert result.held_out_rows >= 2 * 5 * rows_per_position
    assert result.held_out_error is not None and 0.0 <= result.held_out_error <= 1.0
    assert any(message.startswith("Distilled ") and f"valued at the {target} target" in message for message in caplog.messages)
