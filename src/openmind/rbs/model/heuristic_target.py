from dataclasses import dataclass

from openmind.knowledge.constant.rule_kind_constant import POSITION
from openmind.knowledge.constant.task_constant import POSITION_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase


@dataclass(frozen=True, slots=True)
class HeuristicTarget:
    """Where a producer of heuristic rules links what it fits: a ruleset of that context, named as applications
    name it, in that knowledge base.

    **Which task it is a model of, and what kind its rules are, are carried rather than assumed.** They were
    written into the linking as constants, so everything fitted was a position value however it had been
    valued — and a fit over what each *action* is worth had nowhere to land but the position value's ruleset,
    where anything asking for a position value would have found it and believed it. Both names are in closed
    lists already; naming them here is the whole of what was missing.

    Position value is the default because it is what every caller was fitting when this carried nothing."""

    knowledge_base: KnowledgeBase
    context: str
    task: str = POSITION_VALUE
    kind: str = POSITION
