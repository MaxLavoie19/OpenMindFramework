from pathlib import Path

import pytest

from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.model_record import ModelRecord
from openmind.model.constant.model_constant import NETWORK, PROCESSING_TIME, READINGS
from openmind.model.factory.model_factory import create_model_registry
from openmind.model.service.model_timer import ModelTimer
from openmind.timing.service.manual_time_source import ManualTimeSource


def timed_model(tmp_path: Path) -> tuple[ModelTimer, ModelRecord, object]:
    knowledge = create_knowledge_base("timing", tmp_path)
    model = create_model_registry().register(
        knowledge,
        ModelRecord(
            "the network",
            POSITION_VALUE,
            knowledge.ensure_context("chess").id,
            NETWORK,
            knowledge.ensure_mechanism("the network").id,
        ),
    )
    source = ManualTimeSource()
    return ModelTimer(source), model, source


def test_every_reading_is_timed_and_the_mean_kept_with_its_count(tmp_path: Path) -> None:
    timer, model, source = timed_model(tmp_path)
    knowledge = create_knowledge_base("timing", tmp_path)

    for seconds in (0.2, 0.4):
        timer.timed(knowledge, model, lambda seconds=seconds: source.advance(seconds))

    belief = knowledge.belief(PROCESSING_TIME.format(model=model.id), model.context)
    assert belief is not None and float(belief.value) == pytest.approx(0.3)  # type: ignore[arg-type]
    assert dict(belief.tags)[READINGS] == 2


def test_what_a_reading_gives_comes_back(tmp_path: Path) -> None:
    timer, model, _ = timed_model(tmp_path)
    knowledge = create_knowledge_base("timing", tmp_path)

    assert timer.timed(knowledge, model, lambda: "valued") == "valued"
