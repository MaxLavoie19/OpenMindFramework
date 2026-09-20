from typing import Protocol

from openmind.heuristic.model.node import Node
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.search.model.guidance import Guidance
from openmind.search.model.search_settings import SearchSettings
from openmind.search.model.strategy import Strategy


class Planner[Model](Protocol):
    """The planning task: the strategy to follow from this node — a move distribution per state — or None where the
    model has nothing to say here.

    A search is one model of this task, not something OMF enforces: minimax suits a small game, Monte-Carlo tree
    search a large one, semi-determinized Monte-Carlo tree search a game with hidden information, and some work needs
    no search at all, a conversation being improvised rather than planned."""

    def plan(
        self,
        model: Model,
        knowledge_base: KnowledgeBase,
        node: Node,
        guidance: Guidance,
        settings: SearchSettings,
    ) -> Strategy | None: ...
