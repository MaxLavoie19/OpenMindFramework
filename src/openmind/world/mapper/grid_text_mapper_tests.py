from openmind.structure.model.grid import Grid
from openmind.world.mapper.grid_text_mapper import GridTextMapper
from openmind.world.model.state import State


def test_a_two_dimensional_grid_is_laid_out_and_other_models_follow_as_lines() -> None:
    board = Grid.filled((2, 2), None).placed((1, 2), "X")

    assert GridTextMapper().to_text(State.of(cell=board, turn="O")) == "cell 1 2\n   1 . X\n   2 . .\nturn = 'O'"


def test_columns_widen_to_fit_two_digit_numbers() -> None:
    text = GridTextMapper().to_text(State.of(cell=Grid.filled((1, 10), None)))

    assert text.splitlines()[0] == "cell  1  2  3  4  5  6  7  8  9 10"


def test_a_state_without_grids_is_one_line_per_model() -> None:
    assert GridTextMapper().to_text(State.of(turn="O", score=3)) == "score = 3\nturn = 'O'"
