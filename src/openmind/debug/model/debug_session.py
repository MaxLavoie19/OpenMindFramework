from dataclasses import dataclass
from pathlib import Path

from openmind.debug.constant.debug_constant import FREEZE
from openmind.debug.model.breakpoint import Breakpoint
from openmind.debug.model.verbosity import Verbosity


@dataclass(frozen=True, slots=True)
class DebugSession:
    """How a run is logged and debugged. `verbosities` say which log lines are kept, minimal by default. `breakpoints`
    say where to stop. While paused, OMF's clocks freeze or run, and what was measured across a pause is kept for
    training and timing, or left out. Logs go to a file named for the time the session starts in `log_directory`, or to
    `log_file`; with neither, the session writes no file and whoever started it attaches their own handler."""

    name: str
    verbosities: tuple[Verbosity, ...] = (Verbosity(),)
    breakpoints: tuple[Breakpoint, ...] = ()
    clocks_while_paused: str = FREEZE
    keep_paused_measurements: bool = False
    log_directory: Path | None = None
    log_file: Path | None = None
