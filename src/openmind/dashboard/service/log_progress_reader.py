import re
from collections import deque
from pathlib import Path

from openmind.dashboard.constant.dashboard_constant import (
    DEDUCTION_LOGGER,
    GAME_LOGGER,
    MATCH_LOGGER,
    NOTABLE_LOGGERS,
    RECENT_LINES,
    ROUND_LOGGER,
    SEARCH_LOGGER,
)
from openmind.dashboard.model.log_progress import LogProgress
from openmind.dashboard.service.incremental_line_reader import IncrementalLineReader

#: A log line as training logs write it: its level, padded, its logger's name and its message.
LINE = re.compile(r"^(?P<level>[A-Z]+) +(?P<logger>\S+) (?P<message>.*)$")
ROUND_START = re.compile(r"^Round (?P<round>\d+) of (?P<rounds>\d+): (?P<note>.*)$")
#: The payoffs a finished self-play game's line ends with: `payoffs white=0.5 black=0.5`.
PAYOFFS = re.compile(r"payoffs (?P<payoffs>(?:\S+=\S+)(?: \S+=\S+)*)$")


class LogProgressReader:
    """Follows the newest training log of a directory: the round being run, counted from its start line, the self-play
    games and games against opponents finished, the moves searched and the moves deduced since that line, how many of
    those self-play games were drawn and how many decisive, and the latest notable INFO and WARNING lines. It reads only
    what was written since the previous call; a newer log starts over."""

    def __init__(self, line_reader: IncrementalLineReader) -> None:
        self._line_reader = line_reader
        self._path: Path | None = None
        self._round: int | None = None
        self._rounds: int | None = None
        self._note = ""
        self._games = self._matches = self._searched = self._deduced = self._draws = self._decisive = 0
        self._recent: deque[str] = deque(maxlen=RECENT_LINES)

    def progress(self, directory: Path) -> LogProgress | None:
        """The progress of the newest `*.log` in the directory, by modification time; None without one."""
        logs = sorted(directory.glob("*.log"), key=lambda path: path.stat().st_mtime) if directory.is_dir() else []
        if not logs:
            return None
        newest = logs[-1]
        if newest != self._path:
            self._start_over(newest)
        for text in self._line_reader.new_lines(newest):
            self._read(text)
        return LogProgress(
            newest,
            self._round,
            self._rounds,
            self._note,
            self._games,
            self._matches,
            self._searched,
            self._deduced,
            tuple(self._recent),
            self._draws,
            self._decisive,
        )

    def _start_over(self, path: Path) -> None:
        if self._path is not None:
            self._line_reader.forget(self._path)
        self._path, self._round, self._rounds, self._note = path, None, None, ""
        self._games = self._matches = self._searched = self._deduced = self._draws = self._decisive = 0
        self._recent.clear()

    def _count_result(self, message: str) -> None:
        """A draw when every payoff the line ends with is the same number, decisive when they differ."""
        found = PAYOFFS.search(message)
        if found is None:
            return
        try:
            payoffs = {float(item.split("=", 1)[1]) for item in found["payoffs"].split()}
        except ValueError:
            return
        if len(payoffs) == 1:
            self._draws += 1
        else:
            self._decisive += 1

    def _read(self, text: str) -> None:
        line = LINE.match(text)
        if line is None:
            return
        logger, message, level = line["logger"], line["message"], line["level"]
        if logger == GAME_LOGGER and message.startswith("Self-play game ") and PAYOFFS.search(message):
            self._games += 1
            self._count_result(message)
        elif logger == MATCH_LOGGER and message.startswith("Game with seeds "):
            self._matches += 1
        elif logger == SEARCH_LOGGER and message.startswith("Searching "):
            self._searched += 1
        elif logger == DEDUCTION_LOGGER and message.startswith("Deduced "):
            self._deduced += 1
        if logger == ROUND_LOGGER and (start := ROUND_START.match(message)):
            self._round, self._rounds, self._note = int(start["round"]), int(start["rounds"]), start["note"]
            self._games = self._matches = self._searched = self._deduced = self._draws = self._decisive = 0
        if level in ("INFO", "WARNING") and logger in NOTABLE_LOGGERS:
            self._recent.append(f"{level} {logger.rsplit('.', 1)[-1]}: {message}")
