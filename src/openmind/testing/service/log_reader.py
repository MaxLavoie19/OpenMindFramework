import re
from pathlib import Path

#: What a saved log line looks like: when it was produced, then its level, its logger and what it says.
LINE = re.compile(r"(?P<when>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (?P<said>.*)")


def said(log: Path) -> list[str]:
    """What each line of a log says, without the time it was produced at; a line without one raises AssertionError.

    Tests read logs to see what a run decided, not when: taking the time off keeps them readable, and checking that
    every line has one keeps the time from going missing unnoticed."""
    lines = log.read_text(encoding="utf-8").splitlines()
    matched = [LINE.fullmatch(line) for line in lines]
    missing = [line for line, found in zip(lines, matched, strict=True) if found is None]
    assert not missing, f"every log line carries the time it was produced at, but {missing[0]!r} does not"
    return [found["said"] for found in matched if found is not None]
