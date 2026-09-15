import pytest

from openmind.training.model.signal import Signal
from openmind.training.model.signal_record import SignalRecord


@pytest.mark.parametrize(
    ("agreements", "disagreements", "accuracy", "reliability"),
    [(0, 0, 0.5, 0.0), (7, 3, 0.7, pytest.approx(0.4)), (3, 7, 0.3, 0.0), (10, 0, 1.0, 1.0)],
)
def test_reliability_is_twice_the_accuracy_less_one_and_never_below_0(
    agreements: int, disagreements: int, accuracy: float, reliability: float
) -> None:
    record = SignalRecord(Signal("pieces"), agreements, disagreements)

    assert (record.accuracy, record.reliability) == (accuracy, reliability)
