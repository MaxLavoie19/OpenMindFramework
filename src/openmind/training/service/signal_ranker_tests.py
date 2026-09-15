from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord
from openmind.training.service.signal_ranker import SignalRanker


def signal(name: str) -> Signal:
    return Signal(name, PythonRule(f"{name}(me)"))


LIBRARY = SignalLibrary(
    "chess",
    (
        SignalRecord(Signal("win"), 50, 0),
        SignalRecord(signal("rarely"), 7, 3),
        SignalRecord(signal("always"), 10, 0),
        SignalRecord(signal("unread"), 0, 0),
        SignalRecord(signal("often"), 70, 30),
    ),
)


def test_the_best_signals_are_the_most_reliable_read_ones_ties_going_to_the_most_read() -> None:
    best = SignalRanker().best(LIBRARY, 3)

    assert [record.signal.name for record in best] == ["always", "often", "rarely"]
    assert SignalRanker().best(LIBRARY, 0) == ()


def test_a_signal_never_read_counts_as_fully_reliable() -> None:
    signals = (Signal("win"), signal("often"), signal("unread"), signal("new"))

    reliabilities = SignalRanker().reliabilities(LIBRARY, signals)

    assert reliabilities == {"win": 1.0, "often": reliabilities["often"], "unread": 1.0, "new": 1.0}
    assert round(reliabilities["often"], 6) == 0.4
