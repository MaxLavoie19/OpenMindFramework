from openmind.world.mapper.grid_text_mapper import GridTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State


def to_text(*variables: tuple[str, object]) -> str:
    return GridTextMapper(VariableNameMapper()).to_text(State(tuple(variables)))  # type: ignore[arg-type]


def test_cells_with_a_row_and_a_column_become_a_grid_and_other_variables_follow_as_lines() -> None:
    text = to_text(
        ("cell(1,1)", "X"), ("cell(1,2)", None), ("cell(2,1)", None), ("cell(2,2)", "O"), ("payoff(X)", None), ("turn", "O")
    )

    assert text == "cell 1 2\n   1 X .\n   2 . O\npayoff(X) = None\nturn = 'O'"


def test_columns_widen_to_fit_two_digit_numbers() -> None:
    text = to_text(*((f"cell(1,{col})", "X" if col == 10 else None) for col in range(1, 11)))

    assert text == "cell  1  2  3  4  5  6  7  8  9 10\n   1  .  .  .  .  .  .  .  .  .  X"


def test_a_missing_cell_is_left_blank() -> None:
    assert to_text(("cell(1,1)", "X"), ("cell(2,2)", "O")) == "cell 1 2\n   1 X  \n   2   O"


def test_a_state_without_grid_variables_is_one_line_per_variable() -> None:
    assert to_text(("payoff", None), ("turn", "solver")) == "payoff = None\nturn = 'solver'"
