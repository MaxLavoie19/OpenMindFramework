from openmind.timing.model.deadline import Deadline
from openmind.timing.service.manual_time_source import ManualTimeSource


def test_a_deadline_counts_down_on_its_source_and_passes_when_reached() -> None:
    source = ManualTimeSource(100.0)
    deadline = Deadline.after(2.5, source)

    assert (deadline.at, deadline.remaining(), deadline.passed()) == (102.5, 2.5, False)

    source.advance(2.0)
    assert (deadline.remaining(), deadline.passed()) == (0.5, False)

    source.advance(0.5)
    assert deadline.passed()

    source.advance(1.0)
    assert (deadline.remaining(), deadline.passed()) == (-1.0, True)


def test_a_deadline_of_no_time_has_already_passed() -> None:
    assert Deadline.after(0.0, ManualTimeSource()).passed()
