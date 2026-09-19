import itertools
from collections.abc import Callable

import pytest

from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.rbs.service.consequence_library_tests import Declare
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.rule.model.python_rule import PythonRule
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


#: X to act, with X on (1,1) and (1,2): X wins at (1,3).
WIN_IN_1 = "XX./OO./..."
#: X to act, with X on (3,1) and (3,3) and O on (2,3) and (3,2): X can't win at once, but wins within 3 plies from (1,1) or
#: (2,2), which each threaten two lines.
WIN_IN_3 = ".../..O/XOX"
#: O to act, with X on (2,2), (3,1) and (3,3) and O on (2,3) and (3,2): whatever O plays, X wins next.
LOST_IN_2 = ".../.XO/XOX"


def tictactoe(declared: Declare) -> RuleBasedSystem:
    """Tic-tac-toe, declared here: X and O take turns marking an empty cell of a 3 by 3 grid; three in a row wins, a
    full grid is a draw."""
    effects = PythonRule(
        "cell = cell.placed((row, col), turn)\n"
        "if any(all(cell[where] == turn for where in line) for line in cell.lines(3)):\n"
        "    payoff = Map.of({turn: 1.0, 'O' if turn == 'X' else 'X': 0.0})\n"
        "elif None not in cell.cells:\n"
        "    payoff = Map.of({'X': 0.5, 'O': 0.5})\n"
        "turn = 'O' if turn == 'X' else 'X'"
    )
    constraints = (PythonRule("payoff['X'] is None"), PythonRule("cell[row, col] is None"))
    return declared(
        position(".../.../..."),
        legal={"place": constraints},
        outcomes={"place": ((1.0, effects),)},
        players=Players(("X", "O"), "turn", "payoff"),
        parameters={"place": {"row": PythonRule("(1, 2, 3)"), "col": PythonRule("(1, 2, 3)")}},
        empties={"cell": None},
        context="tic-tac-toe",
    )


def position(rows: str, payoff: dict[str, float | None] | None = None) -> State:
    """A tic-tac-toe position from its rows, top first, `.` for an empty cell, no payoff set unless given; X acts when
    both players have as many marks, O otherwise."""
    marks = tuple(None if mark == "." else mark for line in rows.split("/") for mark in line)
    turn = "X" if marks.count("X") == marks.count("O") else "O"
    return State.of(cell=Grid((3, 3), marks), turn=turn, payoff=Map.of(payoff or {"X": None, "O": None}))


def place(row: int, col: int) -> Action:
    return Action("place", (("col", col), ("row", row)))


def new_deducer(clock: Callable[[], float] | None = None) -> PositionDeducer:
    parts = (StateReader(), ActionTextMapper())
    return PositionDeducer(*parts) if clock is None else PositionDeducer(*parts, clock)


def test_a_win_at_once_is_proven_within_one_ply(declared: Declare) -> None:
    deduction = new_deducer().deduce(tictactoe(declared), position(WIN_IN_1), DeductionBudget(1, 60.0))

    assert (deduction.player, deduction.action, deduction.payoffs, deduction.plies, deduction.proven) == (
        "X",
        place(1, 3),
        (1.0, 0.0),
        1,
        True,
    )
    assert [action for action, _ in deduction.line] == [place(1, 3)]


def test_a_forced_win_is_proven_at_the_first_depth_that_reaches_its_end(declared: Declare) -> None:
    deduction = new_deducer().deduce(tictactoe(declared), position(WIN_IN_3), DeductionBudget(3, 60.0))

    assert (deduction.payoffs, deduction.plies) == ((1.0, 0.0), 3)
    assert deduction.action in (place(1, 1), place(2, 2))
    assert len(deduction.line) == 3 and deduction.line[-1][1].model("payoff")["X"] == 1.0


def test_a_position_where_every_move_loses_is_proven_lost(declared: Declare) -> None:
    deduction = new_deducer().deduce(tictactoe(declared), position(LOST_IN_2), DeductionBudget(2, 60.0))

    assert (deduction.player, deduction.payoffs, deduction.plies, len(deduction.line)) == ("O", (1.0, 0.0), 2, 2)


def test_a_win_further_than_the_plies_stays_unproven(declared: Declare) -> None:
    deduction = new_deducer().deduce(tictactoe(declared), position(WIN_IN_3), DeductionBudget(2, 60.0))

    assert (deduction.proven, deduction.action, deduction.payoffs, deduction.line, deduction.plies) == (False, None, None, (), 2)


def test_a_deduction_stops_when_its_seconds_run_out(declared: Declare) -> None:
    times = itertools.chain((0.0,), itertools.repeat(5.0))

    deduction = new_deducer(lambda: next(times)).deduce(tictactoe(declared), position(WIN_IN_3), DeductionBudget(3, 1.0))

    assert (deduction.proven, deduction.plies) == (False, 0)


def test_a_budget_without_plies_or_seconds_or_a_position_without_a_legal_action_raises(declared: Declare) -> None:
    rbs, deducer = tictactoe(declared), new_deducer()
    finished = position(WIN_IN_1, {"X": 1.0, "O": 0.0})

    with pytest.raises(ValueError, match="at least 1 ply"):
        deducer.deduce(rbs, position(WIN_IN_1), DeductionBudget(0, 60.0))
    with pytest.raises(ValueError, match="at least 1 ply"):
        deducer.deduce(rbs, position(WIN_IN_1), DeductionBudget(1, 0.0))
    with pytest.raises(ValueError, match="No legal action"):
        deducer.deduce(rbs, finished, DeductionBudget(1, 60.0))
