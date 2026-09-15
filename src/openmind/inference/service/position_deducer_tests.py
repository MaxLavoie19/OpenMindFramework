import itertools
from collections.abc import Callable

import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
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


def position(rows: str) -> State:
    """A tic-tac-toe position from its rows, top first, `.` for an empty cell, no payoff set; X acts when both players
    have as many marks, O otherwise."""
    cells = {
        f"cell({row},{col})": None if mark == "." else mark
        for row, line in enumerate(rows.split("/"), start=1)
        for col, mark in enumerate(line, start=1)
    }
    marks = [mark for mark in cells.values() if mark is not None]
    turn = "X" if marks.count("X") == marks.count("O") else "O"
    return State(tuple(sorted({**cells, "turn": turn, "payoff(X)": None, "payoff(O)": None}.items())))


def place(row: int, col: int) -> Action:
    return Action("place", (("col", col), ("row", row)))


def new_deducer(clock: Callable[[], float] | None = None) -> PositionDeducer:
    parts = (create_solver(), create_predictor(), StateReader(), ActionTextMapper())
    return PositionDeducer(*parts) if clock is None else PositionDeducer(*parts, clock)


def test_a_win_at_once_is_proven_within_one_ply() -> None:
    deduction = new_deducer().deduce(create_tictactoe_domain(), position(WIN_IN_1), DeductionBudget(1, 60.0))

    assert (deduction.player, deduction.action, deduction.payoffs, deduction.plies, deduction.proven) == (
        "X",
        place(1, 3),
        (1.0, 0.0),
        1,
        True,
    )
    assert [action for action, _ in deduction.line] == [place(1, 3)]


def test_a_forced_win_is_proven_at_the_first_depth_that_reaches_its_end() -> None:
    deduction = new_deducer().deduce(create_tictactoe_domain(), position(WIN_IN_3), DeductionBudget(3, 60.0))

    assert (deduction.payoffs, deduction.plies) == ((1.0, 0.0), 3)
    assert deduction.action in (place(1, 1), place(2, 2))
    assert len(deduction.line) == 3 and dict(deduction.line[-1][1].variables)["payoff(X)"] == 1.0


def test_a_position_where_every_move_loses_is_proven_lost() -> None:
    deduction = new_deducer().deduce(create_tictactoe_domain(), position(LOST_IN_2), DeductionBudget(2, 60.0))

    assert (deduction.player, deduction.payoffs, deduction.plies, len(deduction.line)) == ("O", (1.0, 0.0), 2, 2)


def test_a_win_further_than_the_plies_stays_unproven() -> None:
    deduction = new_deducer().deduce(create_tictactoe_domain(), position(WIN_IN_3), DeductionBudget(2, 60.0))

    assert (deduction.proven, deduction.action, deduction.payoffs, deduction.line, deduction.plies) == (False, None, None, (), 2)


def test_a_deduction_stops_when_its_seconds_run_out() -> None:
    times = itertools.chain((0.0,), itertools.repeat(5.0))

    deduction = new_deducer(lambda: next(times)).deduce(create_tictactoe_domain(), position(WIN_IN_3), DeductionBudget(3, 1.0))

    assert (deduction.proven, deduction.plies) == (False, 0)


def test_a_budget_without_plies_or_seconds_or_a_position_without_a_legal_action_raises() -> None:
    domain, deducer = create_tictactoe_domain(), new_deducer()
    finished = State(tuple(sorted({**dict(position(WIN_IN_1).variables), "payoff(X)": 1.0, "payoff(O)": 0.0}.items())))

    with pytest.raises(ValueError, match="at least 1 ply"):
        deducer.deduce(domain, position(WIN_IN_1), DeductionBudget(0, 60.0))
    with pytest.raises(ValueError, match="at least 1 ply"):
        deducer.deduce(domain, position(WIN_IN_1), DeductionBudget(1, 0.0))
    with pytest.raises(ValueError, match="No legal action"):
        deducer.deduce(domain, finished, DeductionBudget(1, 60.0))
