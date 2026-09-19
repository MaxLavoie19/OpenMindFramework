import json
import logging
from pathlib import Path

from openmind.debug.model.breakpoint import Breakpoint
from openmind.debug.model.debug_session import DebugSession
from openmind.debug.model.target import Target
from openmind.debug.model.verbosity import Verbosity
from openmind.rule.model.python_rule import PythonRule


class SessionJsonMapper:
    """A debug session to and from JSON. Levels are written by name (`"DEBUG"`), conditions as their Python source:

    {"name": "en passant", "verbosities": [{"level": "INFO"}, {"level": "DEBUG", "frame": "evaluation"}],
     "breakpoints": [{"name": "en passant", "frame": "evaluation", "condition": "en_passant is not None"}],
     "clocks_while_paused": "freeze", "keep_paused_measurements": false, "log_directory": "data/log/debug"}"""

    def to_text(self, session: DebugSession) -> str:
        return json.dumps(
            {
                "name": session.name,
                "verbosities": [
                    {
                        "level": logging.getLevelName(verbosity.level),
                        "logger": verbosity.target.logger,
                        "context": verbosity.target.context,
                        "frame": verbosity.target.frame,
                        "tags": [[key, value] for key, value in verbosity.target.tags],
                    }
                    for verbosity in session.verbosities
                ],
                "breakpoints": [
                    {"name": point.name, "frame": point.frame, "condition": None if point.condition is None else point.condition.source}
                    for point in session.breakpoints
                ],
                "clocks_while_paused": session.clocks_while_paused,
                "keep_paused_measurements": session.keep_paused_measurements,
                "log_directory": None if session.log_directory is None else str(session.log_directory),
                "log_file": None if session.log_file is None else str(session.log_file),
            },
            indent=2,
        )

    def from_text(self, text: str) -> DebugSession:
        data = json.loads(text)
        defaults = DebugSession(str(data["name"]))
        verbosities = tuple(
            Verbosity(
                logging.getLevelName(str(item.get("level", "INFO"))),
                Target(
                    item.get("logger"),
                    item.get("context"),
                    item.get("frame"),
                    tuple((str(key), value) for key, value in item.get("tags", ())),
                ),
            )
            for item in data.get("verbosities", ())
        )
        breakpoints = tuple(
            Breakpoint(
                str(item["name"]),
                str(item["frame"]),
                None if item.get("condition") is None else PythonRule(str(item["condition"])),
            )
            for item in data.get("breakpoints", ())
        )
        directory, file = data.get("log_directory"), data.get("log_file")
        return DebugSession(
            defaults.name,
            verbosities or defaults.verbosities,
            breakpoints,
            str(data.get("clocks_while_paused", defaults.clocks_while_paused)),
            bool(data.get("keep_paused_measurements", defaults.keep_paused_measurements)),
            None if directory is None else Path(directory),
            None if file is None else Path(file),
        )

    def read(self, path: Path) -> DebugSession:
        return self.from_text(path.read_text(encoding="utf-8"))
