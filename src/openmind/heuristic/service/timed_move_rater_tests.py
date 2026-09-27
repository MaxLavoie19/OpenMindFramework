from pathlib import Path

import pytest

from openmind.heuristic.model.node import Node
from openmind.heuristic.service.timed_move_rater import TimedMoveRater
from openmind.knowledge.constant.task_constant import MOVE_VALUE
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.knowledge.model.model_record import ModelRecord
from openmind.model.constant.model_constant import NETWORK, PROCESSING_TIME, READINGS
from openmind.model.factory.model_factory import create_model_registry
from openmind.model.service.model_timer import ModelTimer
from openmind.timing.service.manual_time_source import ManualTimeSource
from openmind.world.model.action import Action
from openmind.world.model.state import State

MOVES = (Action("move", (("to", 1),)), Action("move", (("to", 2),)), Action("move", (("to", 3),)))


class Slowly:
    """A rater taking a fixed time however many actions it is given, which is the point being tested."""

    def __init__(self, source: ManualTimeSource, seconds: float) -> None:
        self._source, self._seconds = source, seconds
        self.asked = 0

    def rate(self, model, node, actions, player):  # noqa: ANN001, ANN201
        self.asked += 1
        self._source.advance(self._seconds)
        return tuple(float(number) for number, _ in enumerate(actions))


def timing(tmp_path: Path, seconds: float = 0.3):
    knowledge = create_knowledge_base("timing", tmp_path)
    record = create_model_registry().register(
        knowledge,
        ModelRecord(
            "the rater", (MOVE_VALUE,), knowledge.ensure_context("a game").id, NETWORK,
            knowledge.ensure_mechanism("the rater").id,
        ),
    )
    source = ManualTimeSource()
    inner = Slowly(source, seconds)
    return TimedMoveRater(inner, ModelTimer(source), knowledge, record), inner, knowledge, record


def test_what_the_rater_says_about_each_action_is_what_comes_back(tmp_path: Path) -> None:
    rater, _, _, _ = timing(tmp_path)

    assert rater.rate(object(), Node(State.of()), MOVES, "white") == (0.0, 1.0, 2.0)


def test_rating_every_action_at_once_is_one_reading_however_many_there_are(tmp_path: Path) -> None:
    """Timed per action, a model that rates thirty moves would look thirty times dearer than one asked thirty
    times, when they did the same work."""
    rater, _, knowledge, record = timing(tmp_path, seconds=0.3)

    rater.rate(object(), Node(State.of()), MOVES, "white")

    belief = knowledge.belief(PROCESSING_TIME.format(model=record.id), record.context)
    assert belief is not None
    assert dict(belief.tags)[READINGS] == 1, "three actions, one reading"
    assert float(belief.value) == pytest.approx(0.3)  # type: ignore[arg-type]


def test_being_asked_about_nothing_is_still_a_reading(tmp_path: Path) -> None:
    """It cost what it cost, and a planner dividing its budget by what a model costs must be told."""
    rater, _, knowledge, record = timing(tmp_path, seconds=0.2)

    assert rater.rate(object(), Node(State.of()), (), "white") == ()
    belief = knowledge.belief(PROCESSING_TIME.format(model=record.id), record.context)
    assert belief is not None and dict(belief.tags)[READINGS] == 1
