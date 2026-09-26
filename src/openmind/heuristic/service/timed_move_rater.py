from openmind.heuristic.model.move_rater import MoveRater
from openmind.heuristic.model.node import Node
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.service.model_timer import ModelTimer
from openmind.world.model.action import Action


class TimedMoveRater[Model]:
    """A move rater whose every reading is timed, and the time kept as a belief about the model.

    The other half of `TimedPositionValuer`, and for the same reason: what a planner may explore is worked out
    from what its models cost, and nothing was measuring what they cost.

    **One reading, whatever it rates.** A rater is asked about every action at once and answers about all of
    them, so that is one reading and is timed as one. Timing it per action would make a model that rates
    thirty moves look thirty times dearer than one asked thirty times, when they did the same work."""

    def __init__(
        self,
        move_rater: MoveRater[Model],
        model_timer: ModelTimer,
        knowledge_base: KnowledgeBase,
        record: ModelRecord,
    ) -> None:
        self._rater = move_rater
        self._timer = model_timer
        self._knowledge_base = knowledge_base
        self._record = record

    def rate(
        self, model: Model, node: Node, actions: tuple[Action, ...], player: str
    ) -> tuple[float | None, ...]:
        """What each action is worth to the player taking it, timed."""
        return self._timer.timed(
            self._knowledge_base, self._record, lambda: self._rater.rate(model, node, actions, player)
        )
