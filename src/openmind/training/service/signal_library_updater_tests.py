import logging

import pytest

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_generation_result import ValueGenerationResult
from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.played_game import PlayedGame
from openmind.training.model.rule_support import RuleSupport
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord
from openmind.training.service.signal_library_updater import SignalLibraryUpdater

pytestmark = pytest.mark.log_level("INFO")

MOBILITY, PIECES, CENTER = PythonRule("here.mobility(me)"), PythonRule("pieces(me)"), PythonRule("center(me)")


def fitted(*strengths: tuple[PythonRule, float]) -> ValueGenerationResult:
    return ValueGenerationResult(ValueBase("chess", 0.0, 0.0, 1.0, ()), (), None, (), strengths)


def test_a_fitted_signal_replaces_its_strengths_and_the_strengths_of_signals_not_fitted_stay() -> None:
    library = SignalLibrary(
        "chess",
        (SignalRecord(Signal("win"), 10, 0), SignalRecord(Signal("pieces"), 7, 3)),
        (RuleSupport(MOBILITY, (("pieces", 0.2), ("win", 0.5))), RuleSupport(CENTER, (("win", 0.3),))),
    )

    updated = SignalLibraryUpdater().update(library, {"win": fitted((MOBILITY, 0.8), (PIECES, -0.1))})

    assert updated.supports == (
        RuleSupport(MOBILITY, (("pieces", 0.2), ("win", 0.8))),
        RuleSupport(PIECES, (("win", -0.1),)),
    )
    assert updated.records == library.records


def test_standing_weighs_each_strength_by_its_signal_s_reliability_and_a_signal_never_read_counts_fully() -> None:
    library = SignalLibrary("chess", (SignalRecord(Signal("win"), 10, 0), SignalRecord(Signal("pieces"), 7, 3)))
    support = RuleSupport(MOBILITY, (("new", 0.1), ("pieces", -0.5), ("win", 0.2)))

    assert SignalLibraryUpdater().standing(library, support) == pytest.approx(0.1 + 0.4 * 0.5 + 0.2)


def test_the_value_bases_become_this_round_s_fits_the_arms_the_next_round_follows() -> None:
    old, new = ValueBase("chess", 0.1, 0.0, 1.0, ()), ValueBase("chess", 0.2, 0.0, 1.0, ())
    library = SignalLibrary("chess", value_bases=(("win", old), ("pieces", old)))

    updated = SignalLibraryUpdater().update(library, {"win": ValueGenerationResult(new, (), None, ())})

    assert updated.value_bases == (("win", new),)


def test_games_between_arms_add_a_win_a_draw_or_a_loss_to_each_arm_s_record(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    library = SignalLibrary("chess", (SignalRecord(Signal("win"), 10, 0, 2, 1, 1, 0),))
    games = (
        PlayedGame((), (), (), (1.0, 0.0), ("win", "pieces")),
        PlayedGame((), (), (), (0.5, 0.5), ("pieces", "win")),
        PlayedGame((), (), (), (1.0, 0.0), ()),
    )

    scored = SignalLibraryUpdater().score(library, games)

    records = {record.signal.name: (record.games, record.wins, record.draws, record.losses) for record in scored.records}
    assert records == {"win": (4, 2, 2, 0), "pieces": (2, 0, 1, 1)}
    assert "Arm pieces: 0 wins, 1 draws, 1 losses of 2 games, score 0.25" in caplog.messages


def test_a_rule_no_reliable_signal_supports_leaves_the_library(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    library = SignalLibrary("chess", (SignalRecord(Signal("coin"), 5, 5), SignalRecord(Signal("win"), 10, 0)))

    updated = SignalLibraryUpdater().update(library, {"coin": fitted((CENTER, 0.9)), "win": fitted((PIECES, 0.1))})

    assert [support.term for support in updated.supports] == [PIECES]
    assert "Signals support 1 rules after fitting 2 signals; dropped 1 rules no reliable signal supports" in caplog.messages
