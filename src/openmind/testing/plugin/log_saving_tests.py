import re
from pathlib import Path

import pytest

#: What a saved line looks like: when it was produced, then its level, its logger and what it says.
LINE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (?P<rest>.*)")

EXAMPLE = """
import logging

import pytest


def test_logs():
    logging.getLogger("example").debug("detail")
    logging.getLogger("example").info("decision")


@pytest.mark.log_level("INFO")
def test_logs_from_info():
    logging.getLogger("example").debug("detail")
    logging.getLogger("example").info("decision")


def test_silent():
    pass
"""


def test_each_test_saves_its_logs_under_the_run_root(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(test_example=EXAMPLE)

    result = pytester.runpytest("-p", "openmind.testing.plugin.log_saving")

    result.assert_outcomes(passed=3)
    logs = pytester.path / "data" / "log" / "test_example"
    assert said(logs / "test_logs.log") == ["DEBUG example detail", "INFO  example decision"]
    assert said(logs / "test_logs_from_info.log") == ["INFO  example decision"]
    assert not (logs / "test_silent.log").exists()


def said(log: Path) -> list[str]:
    """What each saved line says, once the time it was produced at is taken off; a line without one fails the test."""
    lines = log.read_text(encoding="utf-8").splitlines()
    matched = [LINE.fullmatch(line) for line in lines]
    assert all(matched), f"every line carries its time: {lines}"
    return [line["rest"] for line in matched]  # type: ignore[index]
