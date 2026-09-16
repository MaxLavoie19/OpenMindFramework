import pytest

from openmind.timing.mapper.time_control_text_mapper import TimeControlTextMapper
from openmind.timing.model.time_control import TimeControl


@pytest.mark.parametrize(
    ("text", "control"),
    [
        ("3+2", TimeControl(180.0, 2.0)),
        ("1+0", TimeControl(60.0, 0.0)),
        ("0.5+1", TimeControl(30.0, 1.0)),
        (" 10 + 5 ", TimeControl(600.0, 5.0)),
    ],
)
def test_a_time_control_is_read_as_chess_writes_it(text: str, control: TimeControl) -> None:
    assert TimeControlTextMapper().from_text(text) == control


@pytest.mark.parametrize("text", ["3+2", "1+0", "0.5+1", "90+30"])
def test_a_time_control_is_written_back_as_it_was_read(text: str) -> None:
    mapper = TimeControlTextMapper()

    assert mapper.to_text(mapper.from_text(text)) == text


@pytest.mark.parametrize("text", ["3", "3+", "+2", "3m+2s", "three+two", "0+2", "-1+2"])
def test_anything_else_raises(text: str) -> None:
    with pytest.raises(ValueError):
        TimeControlTextMapper().from_text(text)
