import pytest

from openmind.timing.service.manual_time_source import ManualTimeSource


def test_time_moves_only_when_told_to() -> None:
    source = ManualTimeSource(5.0)

    assert source.now() == source.now() == 5.0

    source.advance(1.5)

    assert source.now() == 6.5


def test_time_can_t_be_moved_back() -> None:
    with pytest.raises(ValueError, match="backwards"):
        ManualTimeSource().advance(-1.0)
