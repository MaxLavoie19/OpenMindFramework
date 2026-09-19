import logging
from collections.abc import Callable

from openmind.debug.factory.debugger_factory import process_debugger
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.constant.model_constant import PROCESSING_TIME, READINGS, TOTAL_SECONDS
from openmind.timing.model.time_source import TimeSource
from openmind.timing.service.wall_time_source import WallTimeSource

logger = logging.getLogger(__name__)


class ModelTimer:
    """Times every reading a model gives and keeps what it measured: the mean seconds a reading takes, with the seconds
    and the readings behind it, as a belief about the model.

    Every reading is timed, not a sample of them: the timings are what a model's performance profile is built from, and
    the profiles are what a build shipping with its parts built in is generated from. A reading a debugger pause spans
    is left out, as the debug session says."""

    def __init__(self, time_source: TimeSource | None = None) -> None:
        self._time_source = WallTimeSource() if time_source is None else time_source

    def timed[T](self, knowledge_base: KnowledgeBase, model: ModelRecord, read: Callable[[], T]) -> T:
        """What the reading gave, its seconds kept."""
        started = self._time_source.now()
        given = read()
        spent = self._time_source.now() - started
        if process_debugger().measurement_kept(started):
            self.spent(knowledge_base, model, spent)
        return given

    def spent(self, knowledge_base: KnowledgeBase, model: ModelRecord, seconds: float) -> Belief:
        """Keeps a reading's seconds: the mean of what the model has taken so far."""
        variable = PROCESSING_TIME.format(model=model.id)
        held = knowledge_base.belief(variable, model.context)
        tags = dict(held.tags) if held is not None else {}
        readings = int(tags.get(READINGS, 0)) + 1  # type: ignore[arg-type]
        total = float(tags.get(TOTAL_SECONDS, 0.0)) + seconds  # type: ignore[arg-type]
        return knowledge_base.believe(
            Belief(
                variable,
                model.context,
                total / readings,
                tags=((READINGS, readings), (TOTAL_SECONDS, total), ("model", model.id)),
            )
        )
