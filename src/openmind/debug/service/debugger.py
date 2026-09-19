import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import ContextManager

from openmind.debug.constant.debug_constant import CONTINUE, FREEZE, LOG_FORMAT, OUT, STEP
from openmind.debug.model.debug_session import DebugSession
from openmind.debug.model.frame import Frame
from openmind.debug.model.pause import Pause
from openmind.debug.model.warning import Warning
from openmind.debug.service.verbosity_filter import VerbosityFilter
from openmind.debug.service.warning_handler import WarningHandler
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.structure.model.grid import Grid
from openmind.structure.model.list import List
from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar
from openmind.structure.model.value import Value
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: What a pause answers with: continue, step to the next frame, or step out of the frame paused in.
type Pauser = Callable[[Pause], str]

#: The data models a breakpoint's condition can use.
STRUCTURES = {"Grid": Grid, "List": List, "Map": Map, "Scalar": Scalar}


class Debugger:
    """One per process: logging as a debug session says, OMF's reasoning stack as services open frames, breakpoints,
    interrupts and pauses, and warnings.

    Without a session, or with one that targets no frame and sets no breakpoint, opening a frame costs one call and does
    nothing. Otherwise every frame is kept on the reasoning stack, each linked to the Python line that opened it, and
    checked against the breakpoints; a pause hands the stack to the pauser — the console in the process that started
    the run, the pipe to that process in a worker — and waits for it to say continue, step or out.

    Pauses are timed, so clocks can freeze while paused and measurements taken across a pause can be left out, as the
    session says."""

    def __init__(self) -> None:
        self._session: DebugSession | None = None
        self._tracking = False
        self._targeted = False
        self._local = threading.local()
        self._pauser: Pauser | None = None
        self._interrupted = False
        self._stepping = False
        self._out_below: int | None = None
        self._paused_seconds = 0.0
        self._pauses: list[tuple[float, float]] = []
        self._handlers: list[logging.Handler] = []
        self._filter: VerbosityFilter | None = None
        self._warnings = WarningHandler(self)
        self._conditions: dict[str, object] = {}
        self._level_before = logging.WARNING

    @property
    def session(self) -> DebugSession | None:
        return self._session

    @property
    def tracking(self) -> bool:
        """Whether frames are kept: a breakpoint, a targeted verbosity, an interrupt or a step asks for them."""
        return self._tracking

    def start(
        self, session: DebugSession, knowledge: KnowledgeBase | None = None, pauser: Pauser | None = None
    ) -> None:
        """Sets logging up as the session says — the root logger's level, a file when the session names one, the filter
        keeping the lines its verbosities want, and the handler turning every OMF WARNING line into a warning, kept in
        the knowledge base when one is given — and arms its breakpoints, pausing through the pauser."""
        self.stop()
        self._level_before = logging.getLogger().level
        self._session = session
        self._pauser = pauser
        self._targeted = bool(session.breakpoints) or any(
            verbosity.target.context is not None or verbosity.target.frame is not None or verbosity.target.tags
            for verbosity in session.verbosities
        )
        self._tracking = self._targeted
        self._filter = VerbosityFilter(session.verbosities, self)
        root = logging.getLogger()
        root.setLevel(min(verbosity.level for verbosity in session.verbosities))
        path = session.log_file
        if path is None and session.log_directory is not None:
            path = session.log_directory / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            self.attach(handler)
        self._warnings.use(knowledge)
        root.addHandler(self._warnings)
        logger.info(
            "Debug session %s: %s; %d breakpoints; clocks %s while paused",
            session.name,
            ", ".join(
                f"{logging.getLevelName(verbosity.level)} for {self._target_text(verbosity.target)}"
                for verbosity in session.verbosities
            ),
            len(session.breakpoints),
            session.clocks_while_paused,
        )

    def attach(self, handler: logging.Handler) -> None:
        """Adds a handler to the root logger, keeping only the lines the session's verbosities want."""
        if self._filter is not None:
            handler.addFilter(self._filter)
        logging.getLogger().addHandler(handler)
        self._handlers.append(handler)

    def stop(self) -> None:
        """Takes the session's handlers off, puts the root logger's level back as the session found it, and forgets the
        session."""
        root = logging.getLogger()
        for handler in self._handlers:
            root.removeHandler(handler)
            handler.close()
        self._handlers.clear()
        root.removeHandler(self._warnings)
        if self._session is not None:
            root.setLevel(self._level_before)
        self._session = None
        self._targeted = self._tracking = False

    # frames

    def frame(
        self,
        kind: str,
        *,
        context: str | None = None,
        state: State | None = None,
        details: Mapping[str, Value] | None = None,
        tags: tuple[tuple[str, Value], ...] = (),
    ) -> ContextManager[Frame | None]:
        """Opens a reasoning frame for the step about to run; a no-op without a session that needs frames."""
        if not self._tracking:
            return nullcontext()
        caller = sys._getframe(1)
        location = f"{caller.f_code.co_filename}:{caller.f_lineno}"
        opened = Frame(kind, context, state, tuple(sorted((details or {}).items())), tags, location)
        return self._opened(opened)

    def stack(self) -> tuple[Frame, ...]:
        """The reasoning stack in this thread, innermost last."""
        return tuple(self._frames())

    def interrupt(self) -> None:
        """Pauses at the next frame that opens, whatever the session targets."""
        self._interrupted = True
        self._tracking = True

    def answer(self, pause: Pause) -> str:
        """Takes a pause a worker process sent: hands it to this process's pauser and gives back its answer; continue
        where there is none."""
        if self._pauser is None:
            logger.warning("A worker paused (%s), but nothing can take the pause here; it continues", pause.reason)
            return CONTINUE
        started = time.monotonic()
        answer = self._pauser(pause)
        ended = time.monotonic()
        self._paused_seconds += ended - started
        self._pauses.append((started, ended))
        return answer

    # time

    def paused(self) -> bool:
        return getattr(self._local, "pausing", False)

    def frozen_seconds(self) -> float:
        """How long OMF's clocks were frozen by pauses, 0 where the session lets them run."""
        session = self._session
        return self._paused_seconds if session is not None and session.clocks_while_paused == FREEZE else 0.0

    def measurement_kept(self, started: float) -> bool:
        """Whether a measurement started at that `time.monotonic` moment counts for training and timing: always, unless a
        pause ended since and the session leaves out what was measured across pauses."""
        session = self._session
        if session is None or session.keep_paused_measurements:
            return True
        return not any(ended >= started for _, ended in self._pauses)

    # warnings

    def warn(self, warning: Warning) -> None:
        """Logs the warning at WARNING; the warning handler keeps it."""
        logging.getLogger(__name__).warning("%s: %s", warning.kind, warning.message, extra={"debugger_warning": warning})

    def warnings(self) -> tuple[Warning, ...]:
        return self._warnings.kept()

    # inside

    def _frames(self) -> list[Frame]:
        frames = getattr(self._local, "frames", None)
        if frames is None:
            frames = self._local.frames = []
        return frames

    @contextmanager
    def _opened(self, opened: Frame) -> Iterator[Frame]:
        frames = self._frames()
        frames.append(opened)
        try:
            reason = self._stop_reason(opened, len(frames))
            if reason is not None:
                self._pause(reason)
            yield opened
        finally:
            frames.pop()

    def _stop_reason(self, opened: Frame, depth: int) -> str | None:
        if self._interrupted:
            self._interrupted = False
            return "interrupted"
        if self._stepping:
            return "stepped"
        if self._out_below is not None and depth <= self._out_below:
            return "stepped out"
        session = self._session
        if session is None:
            return None
        for point in session.breakpoints:
            if point.frame == opened.kind and self._holds(point.condition, opened):
                return f"breakpoint {point.name}"
        return None

    def _holds(self, condition: object, opened: Frame) -> bool:
        if condition is None:
            return True
        source = condition.source  # type: ignore[attr-defined]
        code = self._conditions.get(source)
        if code is None:
            code = self._conditions[source] = compile(source, f"<breakpoint {source!r}>", "eval")
        names: dict[str, object] = {**STRUCTURES, "frame": opened}
        if opened.state is not None:
            names.update({name: model.value if isinstance(model, Scalar) else model for name, model in opened.state.models})
        try:
            return bool(eval(code, names))  # noqa: S307 - breakpoints are the developer's own Python, like a game's rules
        except Exception:  # noqa: BLE001 - a condition that can't be read here doesn't stop the run
            logger.debug("Breakpoint condition %r can't be read on this frame", source, exc_info=True)
            return False

    def _pause(self, reason: str) -> None:
        pauser = self._pauser
        if pauser is None:
            logger.warning("Would pause (%s), but nothing can take the pause in process %d", reason, os.getpid())
            return
        pause = Pause(reason, self.stack(), "".join(traceback.format_stack()[:-3]), os.getpid())
        started = time.monotonic()
        self._local.pausing = True
        try:
            answer = pauser(pause)
        finally:
            self._local.pausing = False
            ended = time.monotonic()
            self._paused_seconds += ended - started
            self._pauses.append((started, ended))
        self._stepping = answer == STEP
        self._out_below = len(pause.stack) - 1 if answer == OUT else None
        self._tracking = self._targeted or self._stepping or self._out_below is not None
        if answer not in (CONTINUE, STEP, OUT):
            logger.warning("A pause answered %r; continuing", answer)
        logger.info("Paused %.1f seconds (%s); %s", ended - started, reason, answer)

    def _target_text(self, target: object) -> str:
        parts = [
            f"{name} {value}"
            for name, value in (
                ("logger", target.logger),  # type: ignore[attr-defined]
                ("context", target.context),  # type: ignore[attr-defined]
                ("frame", target.frame),  # type: ignore[attr-defined]
            )
            if value is not None
        ]
        if target.tags:  # type: ignore[attr-defined]
            parts.append(f"tags {dict(target.tags)}")  # type: ignore[attr-defined]
        return ", ".join(parts) or "everything"
