import logging

from openmind.budget.model.allocation import Allocation
from openmind.heuristic.model.move_rater import MoveRater
from openmind.heuristic.model.position_valuer import PositionValuer
from openmind.heuristic.service.rule_heuristic import RuleHeuristic
from openmind.heuristic.service.rule_position_valuer import RulePositionValuer
from openmind.model.constant.model_constant import RULES
from openmind.knowledge.constant.task_constant import MOVE_VALUE, POSITION_VALUE
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.model.rule_based_system import RuleBasedSystem
from openmind.search.model.guidance import Guidance

logger = logging.getLogger(__name__)


class Outfitter:
    """Gives a planner the models it is to run with: what the time management policy chose for each task, loaded and
    paired with the service that runs it.

    Pondering a game's rules is worth nothing if the heuristics it deduced sit in the knowledge base unread. This is
    what reads them back: the allocation names a model per task, and each one becomes something the search can ask.

    A task whose chosen model is of a family nothing here can load is left out, and the search does without it rather
    than failing: a model is one way of filling a task, not the only one."""

    def __init__(self, rule_heuristic: RuleHeuristic) -> None:
        self._heuristic = rule_heuristic
        self._valuer = RulePositionValuer(rule_heuristic)

    def outfit(self, knowledge_base: KnowledgeBase, guidance: Guidance, allocation: Allocation) -> Guidance:
        """The guidance with the allocation's models in it, leaving whatever the caller filled itself alone: a caller
        that passed an opponent's model keeps it, and the rest is what was chosen."""
        position_value = guidance.position_value or self._filled(
            knowledge_base, allocation.of(POSITION_VALUE), self._valuer
        )
        move_value = guidance.move_value or self._filled(knowledge_base, allocation.of(MOVE_VALUE), self._heuristic)
        if position_value is not guidance.position_value or move_value is not guidance.move_value:
            logger.debug(
                "Playing with %s",
                ", ".join(
                    f"{task} by {record.name}"
                    for task, record in ((POSITION_VALUE, allocation.of(POSITION_VALUE)), (MOVE_VALUE, allocation.of(MOVE_VALUE)))
                    if record is not None
                )
                or "no heuristic",
            )
        return Guidance(
            guidance.player, position_value, move_value, guidance.agents, guidance.hypotheses, guidance.policies
        )

    def filled(
        self, knowledge_base: KnowledgeBase, record: ModelRecord | None
    ) -> tuple[object, PositionValuer | MoveRater] | None:
        """The model loaded with the service that runs it, by the task it is a model of: what a caller pinning one
        heuristic rather than letting the policy choose needs. None where it is of neither task, or of a family this
        doesn't load."""
        if record is None:
            return None
        if POSITION_VALUE in record.tasks:
            return self._filled(knowledge_base, record, self._valuer)
        return self._filled(knowledge_base, record, self._heuristic) if MOVE_VALUE in record.tasks else None

    def _filled(
        self, knowledge_base: KnowledgeBase, record: ModelRecord | None, service: PositionValuer | MoveRater
    ) -> tuple[object, PositionValuer | MoveRater] | None:
        """The model loaded, with the service that runs it; None where there is none, or where it isn't one of the
        families this loads."""
        model = self._loaded(knowledge_base, record)
        return None if model is None else (model, service)

    def _loaded(self, knowledge_base: KnowledgeBase, record: ModelRecord | None) -> RuleBasedSystem | None:
        """A rule-based model built from the ruleset it was registered by; None for anything else."""
        if record is None or record.family != RULES:
            return None
        ruleset = knowledge_base.ruleset_by_id(record.location)
        context = knowledge_base.context_by_id(record.context)
        if ruleset is None or context is None:
            logger.warning("Model %s names a ruleset that isn't there any more", record.name)
            return None
        return RuleBasedSystem(context.name, context.id, ruleset, knowledge_base.ruleset_rules(ruleset.id))
