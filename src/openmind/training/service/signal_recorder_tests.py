import logging

import pytest

from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.inference.model.deduction import Deduction
from openmind.rbs.service.term_evaluator_tests import new_evaluator
from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.service.position_ponderer_tests import X_WINS, played
from openmind.training.service.signal_recorder import SignalRecorder

pytestmark = pytest.mark.log_level("INFO")

MARKS = Signal("marks", PythonRule("sum(1 for at in here.cell if here.cell[at] == me)"))
WIN = Signal("win")
UNREADABLE = Signal("nothing", PythonRule("here.nothing"))
DRAWN = ((2, 2), (1, 1), (1, 2), (3, 2), (2, 1), (2, 3), (1, 3), (3, 1), (3, 3))


def records(library: SignalLibrary) -> dict[str, tuple[int, int]]:
    return {record.signal.name: (record.agreements, record.disagreements) for record in library.records}


def test_every_position_of_a_decisive_game_is_an_anchor_and_a_drawn_game_adds_nothing(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    domain = create_tictactoe_domain()
    # X wins: before each move X has as many marks as O, or one more; one more points to X.
    signals = (MARKS, WIN, Signal("uniform", parts=("marks", "win")), UNREADABLE)

    library = SignalRecorder(new_evaluator()).record(domain, SignalLibrary("tictactoe"), signals, (played(*X_WINS), played(*DRAWN)))

    assert records(library) == {"marks": (2, 0), "win": (5, 0), "uniform": (5, 0), "nothing": (0, 0)}
    assert (
        "Recorded 4 signals on 5 anchors, 0 of them proven, from 1 decisive games of 2; 1 signals couldn't be read"
        in caplog.messages
    )


def test_a_weighted_aggregation_votes_with_its_parts_by_their_weights() -> None:
    domain, recorder = create_tictactoe_domain(), SignalRecorder(new_evaluator())
    # Reading O's marks for X points to O wherever the players' marks differ: 2 disagreements in X's winning game.
    against = Signal("against", PythonRule("sum(1 for at in here.cell if here.cell[at] == other)"))
    readings = recorder.read(domain, (MARKS, against), (played(*X_WINS),))

    trusting_marks = recorder.aggregate(readings, Signal("weighted", parts=("marks", "against")), {"marks": 1.0, "against": 0.5})
    trusting_against = recorder.aggregate(readings, Signal("weighted", parts=("marks", "against")), {"marks": 0.5, "against": 1.0})

    assert records(recorder.add(SignalLibrary("tictactoe"), (MARKS, against), readings)) == {"marks": (2, 0), "against": (0, 2)}
    assert list(trusting_marks.signs["weighted"]) == [0, 1, 0, 1, 0]  # type: ignore[arg-type]
    assert list(trusting_against.signs["weighted"]) == [0, -1, 0, -1, 0]  # type: ignore[arg-type]


def test_a_blank_reading_is_a_player_lacking_what_the_signal_reads() -> None:
    # Marks once a player has two, blank before: X has two before its fourth and fifth moves, O before the fifth only.
    many = Signal("many", PythonRule("n if (n := sum(1 for at in here.cell if here.cell[at] == me)) >= 2 else None"))

    library = SignalRecorder(new_evaluator()).record(create_tictactoe_domain(), SignalLibrary("tictactoe"), (many,), (played(*X_WINS),))

    assert records(library) == {"many": (1, 0)}


def test_records_keep_counting_across_rounds() -> None:
    domain, recorder, games = create_tictactoe_domain(), SignalRecorder(new_evaluator()), (played(*X_WINS),)

    once = recorder.record(domain, SignalLibrary("tictactoe"), (MARKS,), games)
    twice = recorder.record(domain, once, (MARKS,), games)

    assert records(twice) == {"marks": (4, 0)}


def test_a_proof_names_the_winner_and_a_proven_draw_is_no_anchor() -> None:
    domain, drawn = create_tictactoe_domain(), played(*DRAWN)
    won_by_o = Deduction(drawn.states[3], "X", None, (0.0, 1.0), (), 1)
    drawn_proof = Deduction(drawn.states[4], "O", None, (0.5, 0.5), (), 1)

    library = SignalRecorder(new_evaluator()).record(domain, SignalLibrary("tictactoe"), (WIN,), (drawn,), (won_by_o, drawn_proof))

    assert records(library) == {"win": (1, 0)}


def test_a_domain_without_two_players_raises() -> None:
    from dataclasses import replace

    from openmind.world.model.players import Players

    domain = create_tictactoe_domain()
    alone = replace(domain, players=Players(("X",), "turn", ("payoff(X)",)))

    with pytest.raises(ValueError, match="between two players, not 1"):
        SignalRecorder(new_evaluator()).record(alone, SignalLibrary("tictactoe"), (WIN,), ())
