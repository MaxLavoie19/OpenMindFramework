import logging
import threading
from collections.abc import Callable

from openmind.agent.model.delegation import Delegation
from openmind.agent.model.report import Report
from openmind.agent.service.level import Level
from openmind.heuristic.model.node import Node
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Delegated:
    """A child level running alongside its parent, in a thread of its own: the parent keeps its own seconds and goes
    on while the child pursues what it was given.

    What it costs isn't only seconds — a child running alongside also takes a core the parent could have used — and
    nothing here weighs cores."""

    def __init__(
        self,
        level: Level,
        knowledge_base: KnowledgeBase,
        delegation: Delegation,
        informing: Callable[[Node], State] | None = None,
    ) -> None:
        self._level = level
        self._knowledge_base = knowledge_base
        self._delegation = delegation
        self._informing = informing
        self._report: Report | None = None
        self._stopping = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._running, name=f"level {level.context}", daemon=True)

    def start(self) -> "Delegated":
        """Runs the child in its thread; giving it back so a caller can hold on to it."""
        self._thread.start()
        return self

    def running(self) -> bool:
        """Whether the child is still pursuing its goal."""
        return self._thread.is_alive()

    def report(self, seconds: float | None = None) -> Report | None:
        """What the child gave back, waiting up to those seconds for it; None while it is still running. Without
        seconds it waits as long as the child takes."""
        self._done.wait(seconds)
        return self._report

    def stop(self) -> Report | None:
        """Asks the child to stop and waits for what it did up to then."""
        self._stopping.set()
        self._thread.join()
        return self._report

    def _running(self) -> None:
        try:
            self._report = self._level.run(self._knowledge_base, self._delegation, self._stopping, self._informing)
        finally:
            self._done.set()
