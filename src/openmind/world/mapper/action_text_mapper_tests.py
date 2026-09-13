import pytest

from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action


@pytest.mark.parametrize(
    ("action", "text"),
    [
        (Action("pass", ()), "pass()"),
        (Action("place", (("col", 3), ("row", 2))), "place(col=3, row=2)"),
        (Action("say", (("word", "hi"),)), "say(word='hi')"),
    ],
)
def test_to_text(action: Action, text: str) -> None:
    assert ActionTextMapper().to_text(action) == text
