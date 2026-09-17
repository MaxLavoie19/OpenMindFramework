import pytest

from openmind.agent.mapper.game_summary_json_mapper import GameSummaryJsonMapper
from openmind.agent.model.game_summary import GameSummary
from openmind.agent.model.model_description import ModelDescription
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_control import TimeControl
from openmind.world.model.action import Action

MODEL = ModelDescription("round 2", '{"iterations": 100}')
RANDOM = ModelDescription("random", '{"policy": "uniformly random legal actions"}')


def summary() -> GameSummary:
    return GameSummary(
        "chess",
        "match",
        None,
        3,
        (7, 8),
        ("white", "black"),
        (MODEL, RANDOM),
        (0.0, 1.0),
        41,
        "white's flag",
        None,
        TimeControl(180.0, 2.0),
        (4.0, 0.1, 5.2),
        (None, None, None),
        (Clock(-0.4, 2.0, True), Clock(181.9, 2.0)),
        "white",
        (Action("move", (("uci", "e2e4"),)), Action("move", (("uci", "e7e5"),))),
    )


def test_a_summary_reads_back_as_it_was_written() -> None:
    mapper = GameSummaryJsonMapper()

    assert mapper.from_json(mapper.to_json(summary()), (MODEL, RANDOM)) == summary()


def test_a_summary_keeps_each_model_by_name_and_id_and_its_time_control_as_chess_writes_one() -> None:
    text = GameSummaryJsonMapper().to_json(summary())

    assert f'"model": "round 2", "model_id": "{MODEL.id}"' in text
    assert '"time_control": "3+2"' in text
    assert summary().label == "match game 3"


def test_reading_a_summary_back_without_its_models_raises() -> None:
    mapper = GameSummaryJsonMapper()

    with pytest.raises(ValueError, match="isn't among the models given"):
        mapper.from_json(mapper.to_json(summary()), (MODEL,))
