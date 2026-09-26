from openmind.heuristic.model.node import Node
from openmind.heuristic.model.position_valuer import PositionValuer
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.service.model_timer import ModelTimer


class TimedPositionValuer[Model]:
    """A position valuer whose every reading is timed, and the time kept as a belief about the model.

    **Nothing was timing anything, and a planner was spending its seconds by guesswork.** `PlainTimeManager`
    turns a budget of seconds into a count of nodes by dividing by what the models it will read have been
    costing, and `ModelTimer` writes exactly what it reads — but nothing ever called the timer, so every model
    cost the unmeasured default of a millisecond. Measured against chess that was out by three hundred times:
    a heuristic of sixty-eight terms, some of them looking a move ahead, costs about a third of a second a
    node, so two seconds of intended thinking bought ten minutes of actual thinking and no game ever finished.

    **Every reading, not a sample.** The timings are what a model's profile is built from, and a profile built
    from some of the readings is a profile of a model nobody ran. What it costs to time a reading is one
    subtraction against a reading that already costs milliseconds.

    It reads what it wraps and keeps nothing of its own beyond what it was built with: the model it times, the
    knowledge base the measurement goes to, and the valuer doing the work."""

    def __init__(
        self,
        position_valuer: PositionValuer[Model],
        model_timer: ModelTimer,
        knowledge_base: KnowledgeBase,
        record: ModelRecord,
    ) -> None:
        self._valuer = position_valuer
        self._timer = model_timer
        self._knowledge_base = knowledge_base
        self._record = record

    def values(self, model: Model, node: Node) -> tuple[float, ...] | None:
        """What the position is worth to each player, timed."""
        return self._timer.timed(self._knowledge_base, self._record, lambda: self._valuer.values(model, node))
