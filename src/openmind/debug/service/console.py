from collections.abc import Callable

from openmind.debug.constant.debug_constant import BREAKPOINTS, CONTINUE, HELP, OUT, RESUMING, STACK, STATE, STEP
from openmind.debug.model.breakpoint import Breakpoint
from openmind.debug.model.pause import Pause
from openmind.structure.model.scalar import Scalar

COMMANDS = {
    CONTINUE: "run on to the next breakpoint",
    STEP: "stop at the next frame that opens",
    OUT: "stop at the next frame opening outside this one",
    STATE: "show the paused frame's state",
    STACK: "show the reasoning stack, each frame with the Python line that opened it, then the Python stack",
    BREAKPOINTS: "list the breakpoints",
    HELP: "list these commands",
}


class Console:
    """The terminal side of a pause: shows why the run stopped and where, then reads commands until one resumes it —
    continue, step or out. Reading and writing are given, so a test or another front end can drive it."""

    def __init__(
        self,
        breakpoints: tuple[Breakpoint, ...] = (),
        read: Callable[[str], str] = input,
        write: Callable[[str], None] = print,
    ) -> None:
        self._breakpoints = breakpoints
        self._read = read
        self._write = write

    def __call__(self, pause: Pause) -> str:
        innermost = pause.stack[-1] if pause.stack else None
        where = "no frame" if innermost is None else f"{innermost.kind} at {innermost.python}"
        self._write(f"Paused in process {pause.process} ({pause.reason}): {where}")
        while True:
            try:
                command = self._read("debug> ").strip()
            except EOFError:
                return CONTINUE
            if command in RESUMING:
                return command
            if command == STATE:
                self._write(self._state(pause))
            elif command == STACK:
                self._write(self._stack(pause))
            elif command == BREAKPOINTS:
                self._write(self._listed())
            else:
                self._write("\n".join(f"{name}: {meaning}" for name, meaning in COMMANDS.items()))

    def _listed(self) -> str:
        lines = [
            f"{point.name}: {point.frame}" + (f" where {point.condition.source}" if point.condition else "")
            for point in self._breakpoints
        ]
        return "\n".join(lines) or "No breakpoint"

    def _state(self, pause: Pause) -> str:
        innermost = pause.stack[-1] if pause.stack else None
        if innermost is None or innermost.state is None:
            return "The paused frame has no state"
        return "\n".join(
            f"{name} = {model.value!r}" if isinstance(model, Scalar) else f"{name} = {model!r}"
            for name, model in innermost.state.models
        )

    def _stack(self, pause: Pause) -> str:
        reasoning = [
            f"{depth}. {frame.kind}"
            + (f" in {frame.context}" if frame.context else "")
            + (f" {dict(frame.details)}" if frame.details else "")
            + f" — {frame.python}"
            for depth, frame in enumerate(pause.stack, start=1)
        ]
        return "\n".join(["Reasoning stack, outermost first:", *reasoning, "Python stack:", pause.python_stack.rstrip()])
