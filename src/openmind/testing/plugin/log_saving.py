"""A pytest plugin saving each test's logs, for OpenMind and for the projects built on it. A project loads it from its
root conftest.py: pytest_plugins = ["openmind.testing.plugin.log_saving"]."""

import logging
import sys
from collections.abc import Iterator
from logging.handlers import BufferingHandler

import pytest

from openmind.testing.constant.testing_constant import LOG_DIRECTORY, LOG_FORMAT, LOG_LEVEL_MARKER


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{LOG_LEVEL_MARKER}(level): lowest level of the logs this test saves in {'/'.join(LOG_DIRECTORY)} (default DEBUG)",
    )


@pytest.fixture(autouse=True)
def save_logs_in_data(request: pytest.FixtureRequest) -> Iterator[None]:
    """Saves each test's logs in <root>/data/log/<test file>/<test name>.log, <root> being the test run's root
    directory, from DEBUG up unless marked log_level; a test that logs nothing saves no file."""
    marker = request.node.get_closest_marker(LOG_LEVEL_MARKER)
    handler = BufferingHandler(capacity=sys.maxsize)
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(marker.args[0] if marker else logging.DEBUG)
    try:
        yield
    finally:
        root.setLevel(level)
        root.removeHandler(handler)
        if handler.buffer:
            run_root = request.config.rootpath
            test_file = request.node.path.relative_to(run_root).with_suffix("")
            path = run_root.joinpath(*LOG_DIRECTORY) / test_file / f"{request.node.name.replace('/', '_')}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            formatter = logging.Formatter(LOG_FORMAT)
            path.write_text("".join(f"{formatter.format(record)}\n" for record in handler.buffer), encoding="utf-8")
        handler.close()
