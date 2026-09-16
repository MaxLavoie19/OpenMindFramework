import logging
import math

from openmind.doxastic.constant.doxastic_constant import MEMORY_CHECK_INTERVAL
from openmind.doxastic.model.record import Record
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.service.memory_evictor import evict_oldest
from openmind.parallel.service.memory_meter import MemoryMeter

logger = logging.getLogger(__name__)


class RecallCache:
    """The records in context, held in memory so looking them up again costs nothing.

    Everything remembered goes in, and so does everything attending to a subject, a name or a keyword brings up: while
    the agent works on something, what it knows about it is at hand. Like the reading cache, it empties whenever this
    process holds more than its share of memory and whenever the memory guard asks, so attention is cheap but never
    unbounded — a record dropped from it is still in the store, word for word."""

    def __init__(self, memory_meter: MemoryMeter) -> None:
        self._memory_meter = memory_meter
        self._memory_share = math.inf
        self._kept = 0
        self._records: dict[str, Record] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __len__(self) -> int:
        return len(self._records)

    def memory_entries(self) -> int:
        return len(self._records)

    def evict_memory(self, entries: int) -> None:
        evict_oldest(self._records, entries)

    def clear_memory(self) -> None:
        self.clear()

    def limit_memory(self, memory_bytes: int) -> None:
        """The share of memory this process holds before the records in context are let go."""
        self._memory_share = memory_bytes

    def clear(self) -> None:
        """Lets go of every record held; the store keeps them all."""
        self._records.clear()

    def get(self, record_id: str) -> Record | None:
        return self._records.get(record_id)

    def keep(self, record: Record) -> None:
        """Holds the record in context."""
        self._kept += 1
        if self._kept % MEMORY_CHECK_INTERVAL == 0 and self._memory_meter.resident_bytes() > self._memory_share:
            logger.info("Let go of %d records in context: this process holds more than its share of memory", len(self._records))
            self._records.clear()
        self._memory_guard.remembered()
        self._records[record.id] = record

    def drop(self, record_id: str) -> None:
        """Lets go of one record, such as one forgotten."""
        self._records.pop(record_id, None)
