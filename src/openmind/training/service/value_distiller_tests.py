from collections.abc import Callable
import logging
from pathlib import Path

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.service.game_memory import GameMemory
from openmind.doxastic.factory.knowledge_base_factory import create_knowledge_base
from openmind.rbs.model.value_settings import ValueSettings
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_declarer import RuleDeclarer
from openmind.training.factory.training_factory import create_value_distiller
from openmind.training.model.value_distillation_settings import ValueDistillationSettings

pytestmark = pytest.mark.log_level("INFO")

type Game = Callable[[str], RuleBasedSystem]

VALUES = ValueSettings(prices=(0.1, 0.01), max_steps=200, tolerance=1e-6, seconds=300.0, memory_bytes=1024**3, candidates=1000)


@pytest.mark.parametrize(("target", "rows_per_position"), [("outcome", 2), ("search", 1)])
def test_distill_fits_value_rules_on_self_play_positions_and_measures_them_on_held_out_games(knowledge: KnowledgeBase, tmp_path: Path, game: Game, 
    target: str, rows_per_position: int, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="openmind.training")
    settings = ValueDistillationSettings(games=3, held_out_games=2, iterations=20, seed=1, target=target, values=VALUES)

    result = create_value_distiller(knowledge).distill(game("tictactoe"), AgentBuilder().with_exploration(1.4), settings, RuleDeclarer(knowledge, 'tictactoe'))

    assert result.context == "tictactoe"
    assert [fit.price for fit in result.fits] == [0.1, 0.01]
    assert result.training_rows % rows_per_position == 0 and result.training_rows >= 3 * 5 * rows_per_position
    assert result.held_out_rows >= 2 * 5 * rows_per_position
    assert result.held_out_error is not None and 0.0 <= result.held_out_error <= 1.0
    assert any(message.startswith("Distilled ") and f"valued at the {target} target" in message for message in caplog.messages)


def test_with_a_game_memory_every_self_play_game_is_remembered_under_the_agent_s_name(knowledge: KnowledgeBase, game: Game, tmp_path: Path) -> None:
    memory = GameMemory(create_knowledge_base("tictactoe", tmp_path))
    settings = ValueDistillationSettings(games=3, held_out_games=2, iterations=20, seed=1, target="outcome", values=VALUES)

    create_value_distiller(knowledge, game_memory=memory).distill(game("tictactoe"), AgentBuilder().with_exploration(1.4), settings, RuleDeclarer(knowledge, 'tictactoe'), round_number=2, model_name="round 1"
    )

    assert [model.name for model in memory.models()] == ["round 1"]
    assert memory.scores(("round 1",))["round 1"][0] == 2 * (3 + 2)
