import logging

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.factory.training_factory import create_value_distiller
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.training.model.signal_settings import SignalSettings
from openmind.training.model.value_distillation_settings import ValueDistillationSettings

pytestmark = pytest.mark.log_level("INFO")

VALUES = ValueSettings(prices=(0.1, 0.01), max_steps=200, tolerance=1e-6, seconds=300.0, memory_bytes=1024**3, candidates=1000)


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
    assert result.pondering is None


def test_the_signals_target_records_signals_follows_the_best_and_fits_their_weighted_aggregation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    settings = ValueDistillationSettings(4, 2, 20, 1, "signals", VALUES, None, SignalSettings(arms=2))

    result = create_value_distiller().distill(create_tictactoe_domain(), AgentBuilder().with_exploration(1.4), settings)

    names = [record.signal.name for record in result.arms]
    assert result.library is not None and len(result.library.records) > len(names)
    assert names[0] == "win" and names[-2:] == ["uniform", "weighted"] and 3 <= len(names) <= 5
    assert result.training_rows % 2 == 0 and result.value_base.domain == "tictactoe"
    assert any(message.startswith("Distilled ") and "valued at the signals target" in message for message in caplog.messages)
    assert any(message.startswith("Following weighted: reliability ") for message in caplog.messages)


def test_given_arms_self_play_is_between_them_and_the_round_s_rules_are_the_best_scoring_signal_s(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    settings = ValueDistillationSettings(4, 2, 20, 1, "signals", VALUES, None, SignalSettings(arms=2))
    builders = {arm: AgentBuilder().with_exploration(1.4) for arm in ("win", "uniform", "weighted")}

    result = create_value_distiller().distill(
        create_tictactoe_domain(), AgentBuilder().with_exploration(1.4), settings, None, None, builders
    )

    assert result.library is not None
    games = {record.signal.name: record.games for record in result.library.records if record.games}
    assert sum(games.values()) == 2 * (4 + 2) and set(games) <= set(builders)
    assert sum(1 for message in caplog.messages if message.startswith("Arms game ")) == 4 + 2
    assert any(message.startswith("The round's rules are the ") and " over " in message for message in caplog.messages)


def test_pondering_deduces_the_positions_missed_most_before_fitting(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    pondering = PonderingSettings(2, DeductionBudget(2, 30.0))
    settings = ValueDistillationSettings(3, 2, 20, 1, "outcome", VALUES, pondering)

    result = create_value_distiller().distill(create_tictactoe_domain(), AgentBuilder().with_exploration(1.4), settings)

    assert result.pondering is not None and result.pondering.positions == 2
    assert 0 <= result.pondering.proven <= 2
    assert any(message.startswith("Pondering: 2 positions, ") for message in caplog.messages)
