import logging
import sys
from collections.abc import Iterator
from logging.handlers import BufferingHandler
from pathlib import Path

import pytest

ROOT = Path(__file__).parent


@pytest.fixture(autouse=True)
def save_logs_in_data(request: pytest.FixtureRequest) -> Iterator[None]:
    """Saves each test's logs in data/log/<test file>/<test name>.log, from DEBUG up unless marked log_level."""
    marker = request.node.get_closest_marker("log_level")
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
            test_file = request.node.path.relative_to(ROOT).with_suffix("")
            path = ROOT / "data" / "log" / test_file / f"{request.node.name.replace('/', '_')}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            formatter = logging.Formatter("%(levelname)-5s %(name)s %(message)s")
            path.write_text("".join(f"{formatter.format(record)}\n" for record in handler.buffer), encoding="utf-8")
        handler.close()
