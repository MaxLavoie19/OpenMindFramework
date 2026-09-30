"""Runs the marker wall across rows and window widths through OMF's solver, and says what each cost.

Not a test — it takes long enough that a suite could not hold it. Run it as
`python -m test.integration.marker_wall.marker_wall_sweep [seconds per cell]`.

Capacity is closed form and costs nothing, so every cell is reported whether or not it is solved; a cell the
budget does not reach says so and still shows what it would cover.
"""

import logging
import sys
import time
from pathlib import Path

from openmind.csp.constant.solver_constant import DOMAIN_ORDER
from openmind.csp.factory.csp_factory import create_solver
from openmind.world.model.state import State

# `test` is a package of Python's own, and a real package beats a namespace one wherever it is found, so the
# sibling is imported by its directory rather than through a `test.integration...` path that only pytest resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from marker_wall_circuit import (
    COLORS,
    PAINT,
    alphabet,
    encoding,
    every_pattern,
    positions,
    sequence,
    windows,
)

DEFAULT_SECONDS = 600.0
ROWS = (1, 2, 3)
WINDOWS = (1, 2, 3, 4)

#: What a cell is expected to cost, so one too large is reported rather than started. Domains hold one value per
#: node per out-edge, and a set costs about this many bytes for each.
BYTES_PER_VALUE = 123


def affordable(rows: int, window: int, budget_bytes: float) -> bool:
    return positions(rows, window) * alphabet(rows) * BYTES_PER_VALUE <= budget_bytes


def run(rows: int, window: int) -> tuple[bool, float, int, int, int, int]:
    """Solves one cell and checks it against the closed form. Gives whether it held, the seconds it took, and what
    the search did."""
    names, values, constraints = encoding(rows, window)
    started = time.monotonic()
    found, statistics = create_solver().solve_with_statistics(
        State(()), PAINT, values, constraints, None, limit=1, value_order=DOMAIN_ORDER
    )
    seconds = time.monotonic() - started
    if not found:
        return False, seconds, 0, statistics.assignments, statistics.dead_ends, statistics.pruned_values
    columns = sequence(dict(found[0].parameters), rows, window)
    held = (
        len(columns) == positions(rows, window) + window - 1
        and len(set(windows(columns, window))) == positions(rows, window)
        and set(windows(columns, window)) == every_pattern(rows, window)
    )
    return held, seconds, len(columns), statistics.assignments, statistics.dead_ends, statistics.pruned_values


def main() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SECONDS
    memory = float(sys.argv[2]) if len(sys.argv) > 2 else 8e9
    print(f"{COLORS} colors, one column of markers per metre, {budget:.0f} s and {memory / 1e9:.0f} GB a cell\n")
    print(
        "rows window   positions      metres    columns    markers  solved  checks  assignments  dead ends"
        "  values pruned   seconds"
    )
    for rows in ROWS:
        for window in WINDOWS:
            total = positions(rows, window)
            if not affordable(rows, window, memory):
                print(
                    f"{rows:4d} {window:6d} {total:11d} {total:11d} {total + window - 1:10d}"
                    f" {rows * (total + window - 1):10d}   not reached: too large for the memory budget"
                )
                continue
            held, seconds, columns, assignments, dead_ends, pruned = run(rows, window)
            print(
                f"{rows:4d} {window:6d} {total:11d} {total:11d} {columns:10d} {rows * columns:10d}"
                f" {'yes' if columns else 'no':>7} {'hold' if held else 'FAIL':>7} {assignments:12d}"
                f" {dead_ends:10d} {pruned:14d} {seconds:9.2f}"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
