import logging
from typing import TYPE_CHECKING

from openmind.debug.model.target import Target
from openmind.debug.model.verbosity import Verbosity

if TYPE_CHECKING:
    from openmind.debug.service.debugger import Debugger


class VerbosityFilter(logging.Filter):
    """Keeps a log line when some verbosity wants it: the line is at its level or above, and its target matches — the
    logger by name or prefix, and the context, frame kind and tags against the reasoning frames open when the line is
    written."""

    def __init__(self, verbosities: tuple[Verbosity, ...], debugger: "Debugger") -> None:
        super().__init__()
        self._verbosities = verbosities
        self._debugger = debugger

    def filter(self, record: logging.LogRecord) -> bool:
        return any(
            record.levelno >= verbosity.level and self._matches(verbosity.target, record) for verbosity in self._verbosities
        )

    def _matches(self, target: Target, record: logging.LogRecord) -> bool:
        if target.logger is not None and not (record.name == target.logger or record.name.startswith(f"{target.logger}.")):
            return False
        if target.context is None and target.frame is None and not target.tags:
            return True
        return any(
            (target.context is None or frame.context == target.context)
            and (target.frame is None or frame.kind == target.frame)
            and all(pair in frame.tags for pair in target.tags)
            for frame in self._debugger.stack()
        )
