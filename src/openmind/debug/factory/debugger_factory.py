from openmind.debug.service.debugger import Debugger

_DEBUGGER: Debugger | None = None


def process_debugger() -> Debugger:
    """This process's debugger, made the first time it is asked for."""
    global _DEBUGGER
    if _DEBUGGER is None:
        _DEBUGGER = Debugger()
    return _DEBUGGER
