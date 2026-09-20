import logging
import math
from collections.abc import Sequence

from openmind.budget.model.allocation import Allocation
from openmind.budget.model.budget import Budget
from openmind.heuristic.model.node import Node
from openmind.knowledge.model.model_record import ModelRecord
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.service.model_registry import ModelRegistry
from openmind.search.model.search_settings import SearchSettings

logger = logging.getLogger(__name__)

#: What a node is taken to cost before anything has been measured, in seconds.
UNMEASURED_SECONDS = 0.001

#: The share of a step's seconds the planner may spend; the rest is what reading the models and acting take.
PLANNING_SHARE = 0.8


class PlainTimeManager:
    """The bootstrap time management model: the best measured model of each task, and as many nodes as the seconds
    allow at what a node has been costing.

    Nothing here is learned. It reads what the registry has measured — each model's accuracy and the seconds its
    readings take — and spends what it is given: the planner gets a share of the budget, and the node count is that
    share divided by what the models it will read have been costing. A task nothing has been measured for still gets
    its best model; a task with no model at all is left out, and whatever needed it does without.

    The training step fits a model of this task from what each allocation brought."""

    def __init__(self, model_registry: ModelRegistry) -> None:
        self._registry = model_registry

    def manage(
        self, model: object, knowledge_base: KnowledgeBase, node: Node, budget: Budget, tasks: Sequence[str]
    ) -> Allocation:
        """What to run this step with, and how far to explore."""
        context_id = node.game.context_id if node.game is not None else ""  # type: ignore[union-attr]
        chosen: dict[str, ModelRecord] = {}
        for task in tasks:
            best = self._registry.best(knowledge_base, context_id, task) if context_id else None
            if best is not None:
                chosen[task] = best
        nodes = self._nodes(knowledge_base, chosen, budget)
        logger.debug(
            "%.3g seconds: %d nodes, models %s",
            budget.seconds,
            nodes,
            ", ".join(f"{task} by {record.name}" for task, record in sorted(chosen.items())) or "none",
        )
        return Allocation(SearchSettings(nodes=nodes), chosen)

    def _nodes(self, knowledge_base: KnowledgeBase, chosen: dict[str, ModelRecord], budget: Budget) -> int:
        """How many nodes the seconds allow: the planner's share divided by what reading a node has been costing."""
        seconds = max(budget.seconds, 0.0) * PLANNING_SHARE
        cost = math.fsum(self._seconds(knowledge_base, record) for record in chosen.values()) or UNMEASURED_SECONDS
        return max(int(seconds / cost), 1)

    def _seconds(self, knowledge_base: KnowledgeBase, record: ModelRecord) -> float:
        measured = self._registry.measured(knowledge_base, record).processing_seconds
        return UNMEASURED_SECONDS if measured is None else measured
