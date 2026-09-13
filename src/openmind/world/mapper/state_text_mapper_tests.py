from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.model.state import State


def test_to_text_gives_one_line_per_variable() -> None:
    state = State((("cell(1,1)", "X"), ("payoff(X)", None), ("turn", "O")))

    assert StateTextMapper().to_text(state) == "cell(1,1) = 'X'\npayoff(X) = None\nturn = 'O'"


def test_to_text_of_an_empty_state_is_empty() -> None:
    assert StateTextMapper().to_text(State(())) == ""
