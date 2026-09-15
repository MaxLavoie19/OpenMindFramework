import math

import numpy as np
import pytest
from scipy.special import expit

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.inference.model.deduction import Deduction
from openmind.rbs.service.term_evaluator_tests import new_evaluator
from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.signal import Signal
from openmind.training.service.position_ponderer_tests import X_WINS, played
from openmind.training.service.signal_targeter import SignalTargeter

pytestmark = pytest.mark.log_level("INFO")

MARKS = Signal("marks", PythonRule("sum(1 for at in here.cell if here.cell[at] == me)"))
SIGNALS = (MARKS, Signal("win"), Signal("uniform", parts=("marks", "win")), Signal("weighted", parts=("marks", "win")))
#: X's marks minus O's before each of X's winning game's moves, and the same standardized over both players' rows.
AHEAD = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
MARKS_Z = AHEAD / math.sqrt(0.4)


def test_rows_are_every_position_for_each_player_valued_at_the_final_payoff() -> None:
    game = played(*X_WINS)

    rows = SignalTargeter(new_evaluator()).rows(create_tictactoe_domain(), (game,))

    assert [(row.state, row.player, row.target) for row in rows[:2]] == [(game.states[0], "X", 1.0), (game.states[0], "O", 0.0)]
    assert len(rows) == 2 * len(game.states)


def test_a_signal_s_target_is_expit_of_its_standardized_difference_and_the_aggregations_mix_them() -> None:
    domain, game = create_tictactoe_domain(), played(*X_WINS)

    targets = SignalTargeter(new_evaluator()).targets(domain, (game,), SIGNALS, {"marks": 0.0, "win": 1.0}, 0)

    assert list(targets) == ["marks", "win", "uniform", "weighted"]
    assert targets["marks"][0::2] == pytest.approx(expit(MARKS_Z))
    assert targets["marks"][1::2] == pytest.approx(expit(-MARKS_Z))
    assert (list(targets["win"][0::2]), list(targets["win"][1::2])) == ([1.0] * 5, [0.0] * 5)
    assert targets["uniform"][0::2] == pytest.approx(expit((MARKS_Z + 1.0) / 2))
    assert targets["weighted"][0::2] == pytest.approx(expit(np.ones(5)))


def test_signals_are_read_the_horizon_later_and_a_proof_overrides_every_target() -> None:
    domain, game = create_tictactoe_domain(), played(*X_WINS)
    proof = Deduction(game.states[2], "X", None, (0.0, 1.0), (), 1)

    targets = SignalTargeter(new_evaluator()).targets(domain, (game,), SIGNALS, {}, 1, (proof,))

    later = np.array([1.0, 0.0, 1.0, 0.0, 0.0]) / math.sqrt(0.4)
    expected = expit(later)
    expected[2] = 0.0
    assert targets["marks"][0::2] == pytest.approx(expected)
    assert all(values[4] == 0.0 and values[5] == 1.0 for values in targets.values())


def test_blank_for_both_players_is_no_reading_and_the_aggregations_skip_it() -> None:
    domain, game = create_tictactoe_domain(), played(*X_WINS)
    # Marks once a player has two, blank before: before X's fourth move X has two and O is blank, before the fifth both have two.
    many = Signal("many", PythonRule("n if (n := sum(1 for at in here.cell if here.cell[at] == me)) >= 2 else None"))
    signals = (many, Signal("win"), Signal("uniform", parts=("many", "win")))

    targets = SignalTargeter(new_evaluator()).targets(domain, (game,), signals, {}, 0)

    # The rows with a reading differ by 2, -2, 0 and 0: a spread of √2.
    many_z = np.array([0.0, 0.0, 0.0, math.sqrt(2.0), 0.0])
    assert targets["many"][0::2] == pytest.approx(expit(many_z))
    assert targets["uniform"][0::2] == pytest.approx(expit(np.array([1.0, 1.0, 1.0, (math.sqrt(2.0) + 1.0) / 2, 0.5])))


def test_a_signal_that_can_t_be_read_gets_no_target() -> None:
    domain, game = create_tictactoe_domain(), played(*X_WINS)

    targets = SignalTargeter(new_evaluator()).targets(domain, (game,), (Signal("nothing", PythonRule("here.nothing")), Signal("win")), {}, 0)

    assert list(targets) == ["win"]
