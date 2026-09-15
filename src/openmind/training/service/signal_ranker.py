from collections.abc import Mapping

from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord


class SignalRanker:
    """Ranks a library's signals read from positions, those with a source, by their records."""

    def best(self, library: SignalLibrary, count: int) -> tuple[SignalRecord, ...]:
        """The records of at most count signals with a source and at least one reading, the most reliable first; ties go to
        the signal read most often, then to its name."""
        read = [
            record
            for record in library.records
            if record.signal.source is not None and record.agreements + record.disagreements > 0
        ]
        read.sort(key=lambda record: (-record.reliability, -(record.agreements + record.disagreements), record.signal.name))
        return tuple(read[: max(0, count)])

    def reliabilities(self, library: SignalLibrary, signals: tuple[Signal, ...]) -> Mapping[str, float]:
        """Each signal's reliability by name; a signal never read yet counts as fully reliable, as it does for a rule's
        standing."""
        records = {record.signal.name: record for record in library.records}
        weights: dict[str, float] = {}
        for signal in signals:
            record = records.get(signal.name)
            unread = record is None or record.agreements + record.disagreements == 0
            weights[signal.name] = 1.0 if unread else record.reliability  # type: ignore[union-attr]
        return weights
