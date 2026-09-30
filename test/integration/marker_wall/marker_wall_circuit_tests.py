import pytest

from openmind.csp.constant.solver_constant import DOMAIN_ORDER
from openmind.csp.factory.csp_factory import create_solver
from openmind.world.model.state import State
import sys
from pathlib import Path

# `test` is a package of Python's own, and a real package beats a namespace one wherever it is found, so the
# sibling is imported by its directory rather than through a `test.integration...` path that only pytest resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from marker_wall_circuit import (
    PAINT,
    alphabet,
    encoding,
    every_pattern,
    markers,
    positions,
    sequence,
    windows,
)


def solve(rows: int, window: int) -> list[int]:
    """The wall's columns, through OMF's own solver."""
    names, values, constraints = encoding(rows, window)
    found = create_solver().solve(
        State(()), PAINT, values, constraints, None, limit=1, value_order=DOMAIN_ORDER
    )
    assert found, f"no wall found for {rows} rows and a window of {window}"
    return sequence(dict(found[0].parameters), rows, window)


@pytest.mark.parametrize(("rows", "window"), [(1, 2), (1, 3), (2, 1), (2, 2)])
def test_every_window_of_the_wall_names_one_stretch_of_it(rows: int, window: int) -> None:
    """The closed form says what the answer must be, so it is the oracle rather than whatever the solver does."""
    columns = solve(rows, window)

    assert len(columns) == positions(rows, window) + window - 1
    assert len(windows(columns, window)) == positions(rows, window)
    assert len(set(windows(columns, window))) == positions(rows, window)
    assert set(windows(columns, window)) == every_pattern(rows, window)


def test_a_column_unpacks_to_one_marker_per_row() -> None:
    columns = solve(2, 2)

    assert all(len(markers(column, 2)) == 2 for column in columns)
    assert all(0 <= color < 7 for column in columns for color in markers(column, 2))
    assert {markers(column, 2) for column in columns} == {
        (top, bottom) for top in range(7) for bottom in range(7)
    }


def test_one_row_read_two_columns_at_a_time_covers_every_ordered_pair_of_colors() -> None:
    """Small enough to check by hand: 49 metres of wall, every ordered pair of the seven colors once."""
    columns = solve(1, 2)

    assert len(columns) == 50
    assert sorted(windows(columns, 2)) == sorted((first, second) for first in range(7) for second in range(7))


def test_the_alphabet_is_a_column_of_markers() -> None:
    assert (alphabet(1), alphabet(2), alphabet(3)) == (7, 49, 343)
    assert (positions(2, 2), positions(1, 4), positions(2, 3)) == (2401, 2401, 117649)
