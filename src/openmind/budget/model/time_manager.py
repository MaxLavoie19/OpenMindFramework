from collections.abc import Sequence
from typing import Protocol

from openmind.budget.model.allocation import Allocation
from openmind.budget.model.budget import Budget
from openmind.heuristic.model.node import Node
from openmind.knowledge.service.knowledge_base import KnowledgeBase


class TimeManager[Model](Protocol):
    """The time management task: which models a step runs with, and how much they may explore, given what there is to
    spend.

    It decides from the time available, the models available and what each one trades — its measured accuracy against
    its measured processing time. It works at every level: how many agent models to consider, how much to infer, which
    position and move value models to read, which planner to plan with and how far it may go.

    Planning is worth nothing where guesses can't be educated: with no time, no model or no guiding principle, depth is
    wasted, and this is what decides to go broad or to improvise instead."""

    def manage(
        self, model: Model, knowledge_base: KnowledgeBase, node: Node, budget: Budget, tasks: Sequence[str]
    ) -> Allocation: ...
