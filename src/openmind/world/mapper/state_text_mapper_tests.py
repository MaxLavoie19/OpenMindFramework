from openmind.structure.model.map import Map
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.model.state import State


def test_to_text_gives_one_line_per_model_a_scalar_as_its_value() -> None:
    text = StateTextMapper().to_text(State.of(turn="X", payoff=Map.of({"X": None})))

    assert text == "payoff = Map(items=(('X', None),))\nturn = 'X'"


def test_to_text_of_an_empty_state_is_empty() -> None:
    assert StateTextMapper().to_text(State(())) == ""
