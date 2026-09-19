import argparse
import logging
import signal
from dataclasses import replace
from pathlib import Path

from openmind.debug.factory.debugger_factory import process_debugger
from openmind.debug.mapper.session_json_mapper import SessionJsonMapper
from openmind.debug.model.debug_session import DebugSession
from openmind.debug.model.verbosity import Verbosity
from openmind.debug.service.console import Console
from openmind.debug.service.debugger import Debugger
from openmind.knowledge.service.knowledge_base import KnowledgeBase


def add_debug_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--debug-session",
        type=Path,
        default=None,
        help="a debug session file (JSON): verbosity, breakpoints, and what time does while paused; "
        "Ctrl+C then pauses at the next reasoning frame",
    )


def start_debugging(
    arguments: argparse.Namespace,
    name: str,
    log_directory: Path,
    level: int | str = logging.INFO,
    knowledge: KnowledgeBase | None = None,
) -> Debugger:
    """Starts this process's debugger: with the session file given, its logs in the command's log directory unless it
    names its own; otherwise a session keeping lines at the command's log level. With a session file, the console takes
    pauses, and Ctrl+C interrupts at the next reasoning frame."""
    debugger = process_debugger()
    path = getattr(arguments, "debug_session", None)
    if path is None:
        verbosity = Verbosity(logging.getLevelName(level) if isinstance(level, str) else level)
        session = DebugSession(name, (verbosity,), log_directory=log_directory)
        debugger.start(session, knowledge)
        return debugger
    session = SessionJsonMapper().read(path)
    if session.log_directory is None and session.log_file is None:
        session = replace(session, log_directory=log_directory)
    debugger.start(session, knowledge, Console(session.breakpoints))
    signal.signal(signal.SIGINT, lambda number, frame: debugger.interrupt())
    return debugger
