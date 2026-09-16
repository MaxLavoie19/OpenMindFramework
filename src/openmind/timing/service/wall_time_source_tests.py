import time

from openmind.timing.service.wall_time_source import WallTimeSource


def test_wall_time_moves_on_and_never_back() -> None:
    source = WallTimeSource()

    first = source.now()
    time.sleep(0.01)
    second = source.now()

    assert second - first >= 0.01
