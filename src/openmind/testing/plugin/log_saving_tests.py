import pytest

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
    assert (logs / "test_logs.log").read_text(encoding="utf-8") == "DEBUG example detail\nINFO  example decision\n"
    assert (logs / "test_logs_from_info.log").read_text(encoding="utf-8") == "INFO  example decision\n"
    assert not (logs / "test_silent.log").exists()
