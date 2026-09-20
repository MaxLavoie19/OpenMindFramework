import logging
from collections.abc import Callable, Sequence

from openmind.agent.model.delegation import Delegation
from openmind.agent.model.report import Report
from openmind.agent.service.delegated import Delegated
from openmind.agent.service.level import Level
from openmind.heuristic.model.node import Node
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Hierarchy:
    """The levels OMF runs, by context: a parent delegates by context id and the hierarchy runs that level.

    The contexts are the hierarchy — each one names the level it sits in — so nothing is held here but the levels
    themselves. It keeps nothing else: the knowledge base is given with every call."""

    def __init__(self, levels: Sequence[Level] = ()) -> None:
        self._levels = {level.context: level for level in levels}

    def levels(self) -> tuple[Level, ...]:
        """Every level, in the order they were given."""
        return tuple(self._levels.values())

    def level(self, context: str) -> Level | None:
        """The level of that context, or None where nothing runs there."""
        return self._levels.get(context)

    def children(self, knowledge_base: KnowledgeBase, context: str) -> tuple[Level, ...]:
        """The levels sitting in that one, as their contexts say."""
        return tuple(
            level
            for level in self._levels.values()
            if (held := knowledge_base.context_by_id(level.context)) is not None and held.parent == context
        )

    def perceived(self, node: Node) -> dict[str, State]:
        """Pushes what the shared world now says through every level's abstraction model; what each level sees of it,
        by context."""
        return {context: level.perceived(node) for context, level in self._levels.items()}

    def delegate(self, knowledge_base: KnowledgeBase, delegation: Delegation) -> Report:
        """Runs the child level until it reaches its goal or its budget runs out, and gives back what came of it. The
        parent waits: the seconds were carved out of its own."""
        level = self._level(delegation)
        return level.run(knowledge_base, delegation, informing=self._informing(knowledge_base, level))

    def delegate_alongside(self, knowledge_base: KnowledgeBase, delegation: Delegation) -> Delegated:
        """Runs the child level in a thread of its own and gives back the handle on it. The parent goes on: the
        seconds are the child's own."""
        level = self._level(delegation)
        return Delegated(level, knowledge_base, delegation, self._informing(knowledge_base, level)).start()

    def parent(self, knowledge_base: KnowledgeBase, context: str) -> Level | None:
        """The level of the context this one sits in, or None where nothing runs there."""
        held = knowledge_base.context_by_id(context)
        return None if held is None or held.parent is None else self._levels.get(held.parent)

    def _informing(self, knowledge_base: KnowledgeBase, level: Level) -> Callable[[Node], State] | None:
        """What keeps the parent informed as the child works: the parent perceives where the child stands after every
        step, and what it makes of it is its own abstraction's. None where no level runs above this one."""
        above = self.parent(knowledge_base, level.context)
        return None if above is None else above.perceived

    def _level(self, delegation: Delegation) -> Level:
        level = self._levels.get(delegation.context)
        if level is None:
            raise ValueError(f"No level runs in {delegation.context}")
        logger.debug("Delegating to %s", delegation.context)
        return level
