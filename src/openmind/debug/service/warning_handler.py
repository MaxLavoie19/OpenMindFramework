import logging
from typing import TYPE_CHECKING

from openmind.debug.constant.debug_constant import DEBUGGER, LOGGED
from openmind.debug.model.warning import Warning
from openmind.knowledge.model.direct_experience import DirectExperience
from openmind.knowledge.model.source import Source
from openmind.knowledge.service.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from openmind.debug.service.debugger import Debugger


class WarningHandler(logging.Handler):
    """Turns every OMF WARNING log line into a warning: kept for the viewer and, with a knowledge base, as a direct
    experience of the debugger, so epistemology and tasks can use it. A line the debugger wrote for a warning it was
    given keeps that warning; any other becomes a warning of the kind `logged`."""

    def __init__(self, debugger: "Debugger") -> None:
        super().__init__(logging.WARNING)
        self._debugger = debugger
        self._kept: list[Warning] = []
        self._knowledge: KnowledgeBase | None = None
        self._keeping = False

    def use(self, knowledge: KnowledgeBase | None) -> None:
        self._knowledge = knowledge

    def kept(self) -> tuple[Warning, ...]:
        return tuple(self._kept)

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("openmind") or self._keeping:
            return
        given = getattr(record, "debugger_warning", None)
        warning = given if isinstance(given, Warning) else Warning(LOGGED, record.getMessage(), None, (record.name,))
        self._kept.append(warning)
        knowledge = self._knowledge
        if knowledge is None:
            return
        self._keeping = True
        try:
            context = warning.context or knowledge.ensure_context(DEBUGGER).id
            knowledge.experience(
                DirectExperience(
                    f"warning: {warning.kind}",
                    context,
                    warning.message,
                    Source(knowledge.ensure_mechanism(DEBUGGER).id, (("logger", record.name),), rests_on=warning.ids),
                    tags=(("keyword", "warning"), ("kind", warning.kind)),
                )
            )
        finally:
            self._keeping = False
