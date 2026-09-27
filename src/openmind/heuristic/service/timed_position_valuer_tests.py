from pathlib import Path

import pytest

from openmind.heuristic.model.node import Node
from openmind.heuristic.service.timed_position_valuer import TimedPositionValuer
from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.model_record import ModelRecord
from openmind.model.constant.model_constant import NETWORK, PROCESSING_TIME, READINGS
from openmind.model.factory.model_factory import create_model_registry
from openmind.model.service.model_timer import ModelTimer
from openmind.timing.service.manual_time_source import ManualTimeSource
from openmind.world.model.state import State


class Slowly:
    """A valuer that takes as long as it is told to and says what it is told to say."""

    def __init__(self, source: ManualTimeSource, seconds: float, says: tuple[float, ...] | None) -> None:
        self._source, self._seconds, self._says = source, seconds, says
        self.asked = 0

    def values(self, model: object, node: Node) -> tuple[float, ...] | None:
        self.asked += 1
        self._source.advance(self._seconds)
        return self._says


def timing(tmp_path: Path, seconds: float = 0.25, says: tuple[float, ...] | None = (1.0, -1.0)):
    knowledge = create_knowledge_base("timing", tmp_path)
    record = create_model_registry().register(
        knowledge,
        ModelRecord(
            "the valuer", (POSITION_VALUE,), knowledge.ensure_context("a game").id, NETWORK,
            knowledge.ensure_mechanism("the valuer").id,
        ),
    )
    source = ManualTimeSource()
    inner = Slowly(source, seconds, says)
    return TimedPositionValuer(inner, ModelTimer(source), knowledge, record), inner, knowledge, record


def test_what_the_valuer_says_is_what_comes_back(tmp_path: Path) -> None:
    valuer, _, _, _ = timing(tmp_path)

    assert valuer.values(object(), Node(State.of())) == (1.0, -1.0)


def test_knowing_nothing_is_passed_on_as_knowing_nothing(tmp_path: Path) -> None:
    """None means the model could say nothing here, and timing it must not turn that into a number."""
    valuer, _, _, _ = timing(tmp_path, says=None)

    assert valuer.values(object(), Node(State.of())) is None


def test_the_reading_is_timed_and_kept_as_a_belief_about_the_model(tmp_path: Path) -> None:
    """What a planner divides its budget by. Until this was called every model cost the unmeasured default."""
    valuer, _, knowledge, record = timing(tmp_path, seconds=0.25)

    valuer.values(object(), Node(State.of()))

    belief = knowledge.belief(PROCESSING_TIME.format(model=record.id), record.context)
    assert belief is not None and float(belief.value) == pytest.approx(0.25)  # type: ignore[arg-type]


def test_every_reading_is_timed_and_not_a_sample(tmp_path: Path) -> None:
    """A profile built from some of the readings is a profile of a model nobody ran."""
    valuer, inner, knowledge, record = timing(tmp_path, seconds=0.1)

    for _ in range(3):
        valuer.values(object(), Node(State.of()))

    belief = knowledge.belief(PROCESSING_TIME.format(model=record.id), record.context)
    assert inner.asked == 3
    assert belief is not None and dict(belief.tags)[READINGS] == 3
